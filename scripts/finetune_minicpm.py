"""Fine-tune MiniCPM on the phonetic ASR-evaluation dataset with Modal.

Run with:
    modal run scripts/finetune_minicpm.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "read-along-ai-finetune"
BASE_MODEL_ID = "openbmb/MiniCPM-2B-sft-bf16"
HUB_REPO_ID = "kingkw1/minicpm-phonetic-evaluator"
DATA_PATH = "/root/data/train.jsonl"
CACHE_DIR = "/model-cache"
OUTPUT_DIR = "/tmp/minicpm-phonetic-evaluator"


app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("read-along-ai-minicpm-cache", create_if_missing=True)

finetune_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
    )
    .add_local_file("data/train.jsonl", remote_path=DATA_PATH)
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_DIR}/transformers",
            "TORCH_HOME": f"{CACHE_DIR}/torch",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _format_example(example: dict[str, str], eos_token: str) -> str:
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Input:\n"
        f"{example['input']}\n\n"
        "### Output:\n"
        f"{example['output']}{eos_token}"
    )


@app.function(
    image=finetune_image,
    gpu="A100",
    timeout=1800,
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-write-secret", required_keys=["HF_TOKEN"])],
)
def train_minicpm() -> dict[str, str | int]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    hf_token = _hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN was not found. Create/attach a Modal Secret named "
            "'huggingface-write-secret' that exposes HF_TOKEN."
        )

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    eos_token = tokenizer.eos_token or ""

    def formatting_func(examples):
        return [
            _format_example(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                },
                eos_token=eos_token,
            )
            for instruction, input_text, output in zip(
                examples["instruction"],
                examples["input"],
                examples["output"],
            )
        ]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": lora_config,
        "formatting_func": formatting_func,
        "args": training_args,
    }
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, max_seq_length=512, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)

    trainer.train()

    adapter_dir = Path(OUTPUT_DIR) / "adapter"
    merged_dir = Path(OUTPUT_DIR) / "merged"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del trainer
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    merged_model.push_to_hub(HUB_REPO_ID, token=hf_token, safe_serialization=True)
    tokenizer.push_to_hub(HUB_REPO_ID, token=hf_token)
    model_cache.commit()

    return {
        "base_model": BASE_MODEL_ID,
        "hub_repo": HUB_REPO_ID,
        "train_rows": len(dataset),
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
    }


@app.local_entrypoint()
def main():
    result = train_minicpm.remote()
    print("Fine-tuning complete")
    print(result)
