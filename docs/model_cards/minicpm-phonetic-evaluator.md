---
library_name: transformers
pipeline_tag: text-generation
base_model: openbmb/MiniCPM-2B-sft-bf16
license: other
tags:
  - minicpm
  - phonetic-evaluation
  - reading-assessment
  - child-speech
  - asr-evaluation
  - qlora
  - gguf
  - llama-cpp
  - read-along-ai
---

# MiniCPM Phonetic Evaluator

`kingkw1/minicpm-phonetic-evaluator` is a small binary evaluator for Read-Along AI. Given a target reading item and an ASR transcript, it answers only `True` or `False`: whether the transcript is an acceptable phonetic match for what the child was asked to read.

The model is designed as a second-stage judge after a simple normalized exact-match check. Exact matches are accepted immediately by the app; this model is used for close or ambiguous transcripts, such as plurals, word-boundary splits, or plausible child-speech/ASR substitutions.

## Model Details

- **Developed by:** Kevin King
- **Model ID:** `kingkw1/minicpm-phonetic-evaluator`
- **Model type:** Causal language model fine-tuned for binary text classification through instruction following
- **Base model:** [`openbmb/MiniCPM-2B-sft-bf16`](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)
- **Language:** English
- **Primary task:** Phonetic accept/reject judgment for ASR transcripts in an early-reading app
- **Project repository:** <https://github.com/kingkw1/read-along-ai>
- **Demo Space:** <https://huggingface.co/spaces/build-small-hackathon/read-along-ai>
- **Quantized local artifact:** `minicpm-phonetic-evaluator-q4_k_m.gguf`

## Intended Use

### Direct Use

Use this model to classify whether an ASR transcript preserves the target word or short target sentence closely enough to count as a valid read-aloud attempt. The expected prompt format is:

```text
### Instruction:
Determine if the ASR transcript is a valid phonetic match for the target word. Output only True or False.

### Input:
Target: scientist | ASR: scientists

### Output:
```

The expected output is exactly one boolean token:

```text
True
```

### Downstream Use

In Read-Along AI, the evaluator sits behind an exact normalized string match:

1. Normalize the target text and ASR transcript.
2. Accept exact matches without calling the model.
3. Ask the MiniCPM evaluator only when the transcript is close or ambiguous.
4. Treat unparseable responses as `False`.

This keeps the app fast for obvious correct readings and fail-closed for uncertain cases.

### Out-of-Scope Use

This model is not a general pronunciation scorer, speech therapist, literacy diagnostic, grading system, or safety-critical educational assessment tool. It does not receive audio, phoneme timings, confidence scores, or child-specific context. It should not be used to make high-stakes decisions about reading ability.

## How to Get Started

### Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "kingkw1/minicpm-phonetic-evaluator"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

prompt = """### Instruction:
Determine if the ASR transcript is a valid phonetic match for the target word. Output only True or False.

### Input:
Target: scientist | ASR: scientists

### Output:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = output_ids[0, inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True).strip())
```

### GGUF / llama.cpp

The Read-Along AI local path uses the Q4 GGUF artifact with `llama-cpp-python`. The application resolves the file from `LOCAL_MINICPM_GGUF_PATH`, `models/gguf/minicpm-phonetic-evaluator-q4_k_m.gguf`, or the Hugging Face model repository cache.

## Training Details

### Training Data

The fine-tuning dataset is `data/train.jsonl` in the Read-Along AI project repository. It contains 50 instruction examples derived from a cleaned 50-word child-speech ASR baseline set.

- 38 examples were exact ASR matches and became positive `True` labels.
- 12 strict-match failures were manually reviewed.
- 5 of those failures were labeled acceptable phonetic/ASR variants.
- 7 were labeled wrong-content or insufficient-evidence negatives.
- Final label balance: 43 `True`, 7 `False`.

Example positive variants include:

- `scientist` -> `scientists`
- `sunflower` -> `sunny flowers`
- `window` -> `wind up`

Example negative variants include:

- `sudden` -> `seven`
- `invisible` -> `and where's the ball`
- `pyramid` -> `apparently`

The dataset is intentionally small and task-specific. It was built for a hackathon MVP and should be expanded before broad deployment.

### Training Procedure

Training was run with Modal using `scripts/finetune_minicpm.py` from the project repository.

- **Base model:** `openbmb/MiniCPM-2B-sft-bf16`
- **Trainer:** TRL `SFTTrainer`
- **Fine-tuning method:** 4-bit parameter-efficient training with LoRA
- **Quantization during training:** bitsandbytes NF4, double quantization, bfloat16 compute
- **LoRA rank:** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Target modules:** `q_proj`, `v_proj`
- **Epochs:** 5
- **Batch size:** 2 per device
- **Gradient accumulation:** 4
- **Learning rate:** `2e-4`
- **Optimizer:** `paged_adamw_8bit`
- **Warmup ratio:** 0.03
- **Max sequence length:** 512
- **Training hardware:** Modal A100

After training, the LoRA adapter was merged into the base model and pushed to the Hugging Face model repository as safetensors. A local conversion script, `scripts/convert_to_gguf.py`, then converted the merged model to FP16 GGUF and quantized it to Q4_K_M for `llama.cpp` / `llama-cpp-python` inference.

## Evaluation

The project evaluation is documented in `notebooks/02_post_tuning_evaluation.ipynb`.

Important caveat: this is a small provenance and smoke evaluation on the same 50 examples used to derive the training data, not a held-out benchmark. Treat these numbers as evidence that the integration works and that the model learned the intended boundary on the project examples, not as a generalization claim.

### Results

| Evaluation view | Result |
| --- | ---: |
| Baseline ASR exact-match acceptance | 38/50, 76.0% |
| ASR plus tuned MiniCPM acceptance | 42/50, 84.0% |
| Strict-match failures reviewed by model | 12 |
| Manual-label agreement on strict-match failures | 9/12, 75.0% |
| Overall agreement with manual labels, counting exact matches as true | 47/50, 94.0% |

On the 12 strict-match failures, the tuned evaluator accepted `scientist -> scientists`, `sunflower -> sunny flowers`, `window -> wind up`, and also accepted `guessing -> yeah same`. It rejected several intended negatives correctly, but it missed some manually accepted variants such as `safari -> so far` and `compass -> come this`.

### Interpretation

The model improved the product acceptance path over strict string matching, but the error analysis shows the current dataset is too small and imbalanced. More labeled child-speech and ASR examples are needed, especially negative examples and sentence-level examples, before relying on this model outside the Read-Along AI prototype.

## Bias, Risks, and Limitations

- The model was trained on only 50 examples and may overfit the specific words, ASR system, speaker, and labeling choices.
- Training data is word-level, while the app may also ask the same judge about short sentences. Sentence-level behavior needs more evaluation.
- The model sees text transcripts only. It does not hear the audio and cannot distinguish ASR errors from actual reading errors.
- Dialect, accent, age, articulation patterns, microphone quality, and ASR behavior can all affect transcripts and therefore model decisions.
- False positives can reward an incorrect reading; false negatives can frustrate a child who made a reasonable attempt.
- The base MiniCPM model can be prompt-sensitive. The application should use deterministic generation and parse only `True`/`False`.

## Recommendations

- Use exact normalized matching before calling the model.
- Use deterministic decoding with a very small `max_new_tokens`.
- Parse boolean outputs defensively and fail closed when the response is unclear.
- Keep feedback gentle and low-stakes.
- Add a larger held-out test set before using this beyond prototype or demo settings.
- Prefer human review for curriculum decisions or any high-impact educational assessment.

## Technical Specifications

### Architecture and Objective

The model is a MiniCPM causal language model fine-tuned with instruction-formatted examples. The objective is to produce a single boolean answer for a target/transcript pair.

### Software

The training script pinned:

- `transformers==4.40.2`
- `peft==0.10.0`
- `trl==0.8.6`
- `accelerate==0.29.3`
- `bitsandbytes`
- `datasets`
- `sentencepiece`

The Modal inference endpoint loads the model with `AutoModelForCausalLM` and `AutoTokenizer` using `trust_remote_code=True`. The local offline path uses the Q4_K_M GGUF through `llama-cpp-python`.

## License

The Read-Along AI project code is MIT licensed. The fine-tuned model is derived from `openbmb/MiniCPM-2B-sft-bf16`; use of the model weights is subject to the upstream MiniCPM model license terms. The upstream MiniCPM card states that repository code is Apache-2.0 and that MiniCPM model weights are governed by the General Model License, with academic use allowed and commercial use requiring authorization from ModelBest/OpenBMB.

Review the base model license before redistributing or using this derivative model commercially:

- <https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16>

## Citation

If you use this model, cite the base MiniCPM work as requested by OpenBMB:

```bibtex
@inproceedings{minicpm2024,
  title={MiniCPM: Unveiling the Potential of End-side Large Language Models},
  booktitle={OpenBMB Blog},
  year={2024}
}
```

## Model Card Authors

Kevin King, with drafting support from OpenAI Codex.

## Contact

Hugging Face: <https://huggingface.co/kingkw1>
