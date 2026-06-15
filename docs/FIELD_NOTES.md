---
title: "Building Read-Along AI: Field Notes from a Small-Model Reading Tutor"
published: false
tags: ai, huggingface, gradio, hackathon
---

## The Short Version

Read-Along AI is an offline-capable reading practice app for early readers, built for the Hugging Face Build Small Hackathon.

The app shows one short sentence at a time, lets a child tap individual words for help, records a read-aloud attempt, and gives gentle feedback. Clean readings pass immediately. Close or ambiguous transcripts get a second look from a fine-tuned MiniCPM phonetic evaluator. Meaning-changing mistakes still fail closed.

The submitted Space is here:

https://huggingface.co/spaces/build-small-hackathon/read-along-ai

The repository is here:

https://github.com/kingkw1/read-along-ai

The fine-tuned evaluator model is here:

https://huggingface.co/kingkw1/minicpm-phonetic-evaluator

## Why I Built It

Reading practice is a surprisingly delicate interaction.

A beginning reader needs repetition, encouragement, and immediate feedback. But feedback can easily become too harsh, too delayed, or too dependent on an adult being fully available at every moment. Standard speech recognition also struggles with developing articulation, slow speech, partial confidence, and the small pronunciation differences that are normal for young readers.

The product question behind Read-Along AI was:

Can a small-model app make reading practice feel more like having a patient helper beside the story, without sending a child's voice through a large hosted AI pipeline?

That made the project a natural fit for the Backyard AI track. The intended user is not an abstract enterprise persona. It is a family doing short reading practice at home, where the app needs to be simple, private, and emotionally low-stakes.

## What I Built

The hackathon MVP is intentionally narrow. It focuses on one reliable sentence-reading loop:

1. Show a short curriculum sentence in large, readable text.
2. Let the child tap any word to hear help.
3. Record the child reading the full sentence.
4. Transcribe the audio.
5. Accept exact normalized matches immediately.
6. Ask a fine-tuned MiniCPM evaluator about close or ambiguous transcripts.
7. Celebrate success or ask for a gentle retry.

The current curriculum is small on purpose:

- "The cat sat."
- "The dog ran fast."
- "She had a red hat."
- "I love to play outside."

That scope let me concentrate on the parts that mattered most for the demo: the end-to-end reading loop, the child-facing interface, privacy-preserving local execution, and a more developmentally fair evaluation path than strict string matching.

## Why Strict ASR Was Not Enough

The first baseline was simple: transcribe the child's speech and check whether the transcript exactly matches the target after normalizing casing and punctuation.

That is a good fast path for clean readings, but it is too brittle as the only grading rule. Early readers may speak slowly, substitute sounds, add a plural ending, or produce speech that ASR splits into different words.

Examples from the evaluation set:

- `scientist` -> `scientists`
- `sunflower` -> `sunny flowers`
- `window` -> `wind up`

Those are not all equivalent in a general NLP setting, but for early reading practice they can be plausible evidence that the child attempted the target word.

At the same time, the app must not become a permissive "anything goes" checker. If the sentence is "She had a red hat," then "She had a blue hat" should not pass. The meaning changed.

That led to a two-stage evaluator:

1. Exact normalized match first.
2. MiniCPM phonetic judgment only for close or ambiguous cases.

If the MiniCPM response cannot be parsed confidently, the app fails closed and asks the child to try again.

## The Small-Model Architecture

Read-Along AI uses a dual-mode architecture.

### Turbo Mode

Turbo Mode routes inference through Modal endpoints for lower latency:

- Cohere Transcribe for ASR.
- OpenBMB VoxCPM for text-to-speech and sentence audio.
- A hosted fine-tuned MiniCPM evaluator for close transcript judgment.

Modal also powered the rapid fine-tuning workflow for the MiniCPM evaluator on A100 infrastructure.

### Off the Grid Mode

Off the Grid Mode is the privacy-preserving path. It runs inside the Hugging Face Space process without Modal calls:

- `faster-whisper` handles local ASR.
- The MiniCPM evaluator runs from a Q4 GGUF through `llama-cpp-python`.
- Sentence and word help use committed curriculum WAV files.
- Word clips are sliced from local timing labels rather than generated on demand.

The local path is slower than Turbo Mode, but it demonstrates the core thesis of the hackathon: small, focused models can do useful work without always depending on a frontier-scale cloud API.

## The MiniCPM Phonetic Evaluator

The fine-tuned evaluator is based on `openbmb/MiniCPM-2B-sft-bf16`.

It is trained as a compact binary judge. Given a target reading item and an ASR transcript, it outputs only `True` or `False`: whether the transcript is an acceptable phonetic match for the target.

The training set was intentionally small and task-specific:

- 50 child-speech ASR examples.
- 38 exact ASR matches became positive examples.
- 12 strict-match failures were manually reviewed.
- 5 failures were labeled acceptable phonetic or ASR variants.
- 7 failures were labeled wrong-content or insufficient-evidence negatives.

Fine-tuning used LoRA on Modal, then the merged model was converted to GGUF and quantized for local `llama.cpp` inference.

This model is not a reading diagnostic, speech therapy tool, or general pronunciation scorer. It is a hackathon MVP component that makes the product loop more patient than exact string matching while keeping the final app fail-closed.

## What I Measured

The evaluator notebook is documented in the repository at:

https://github.com/kingkw1/read-along-ai/blob/main/notebooks/02_post_tuning_evaluation.ipynb

The important caveat: these results are a small provenance and smoke evaluation on the same 50 examples used to derive the training data. They show that the app learned the intended boundary on the project examples. They are not a broad generalization claim.

Results:

| Evaluation view | Result |
| --- | ---: |
| Baseline ASR exact-match acceptance | 38/50, 76.0% |
| ASR plus tuned MiniCPM acceptance | 42/50, 84.0% |
| Strict-match failures reviewed by model | 12 |
| Manual-label agreement on strict-match failures | 9/12, 75.0% |
| Overall agreement with manual labels, counting exact matches as true | 47/50, 94.0% |

The product lesson was more important than the headline number: the exact-match path should stay, but a small evaluator can make the experience less punishing when ASR produces plausible near-matches.

## Word Help Was Its Own Mini-Problem

Clickable word help seems simple until you try to make it local.

In Turbo Mode, the app can use TTS. In Off the Grid Mode, I wanted word help to avoid cloud calls. The app therefore commits sentence WAVs and slices word clips from local label timings.

I tried a few approaches for locating word boundaries:

| Method | Manual-gap hits |
| --- | ---: |
| Text-length proportional splitter | 0/13 |
| Earlier faster-whisper alignment path | 1/13 |
| Local signal word-boundary detector | 12/13 |

The final local detector combines a sentence-duration prior with short-time RMS minima, then snaps expected internal boundaries to nearby low-energy frames. On the committed label set, it gets 12 of 13 internal boundaries into the manually labeled silence gaps, with about 5 ms mean boundary error.

That was a useful reminder: sometimes the most robust small solution is not a larger model. It is a focused signal-processing heuristic wrapped around the model output.

## The Interface Had To Stop Looking Like A Notebook

Default Gradio is excellent for building AI demos quickly, but a standard Gradio panel is not the right surface for a 4 to 7 year old reader.

The child-facing UI is custom HTML, CSS, and JavaScript running inside Gradio:

- One large sentence at a time.
- Clickable word spans for help.
- A simple read-aloud control.
- Minimal technical text during the learning loop.
- Progress feedback.
- Confetti and auto-advance on success.

The design goal was not decoration. It was cognitive load reduction. A child should see the sentence, get help, record, and try again. They should not have to parse model logs, API widgets, or data-science controls.

## What Worked

The strongest parts of the build were:

- A narrow MVP scope.
- Exact string matching as a fast path.
- MiniCPM only as a second-stage judge.
- A local GGUF path through `llama.cpp`.
- Modal for fast iteration and Turbo Mode.
- Committed audio assets for a reliable local demo.
- A custom Gradio surface that fits the user instead of exposing the implementation.

The biggest product win is that the app can be forgiving without being careless. "Da dog ran fast" can be treated differently from "The dog ran slow." That distinction matters for a reading tutor.

## What Did Not Work Yet

The current build is still a prototype.

The evaluator was trained on a tiny dataset. It needs a larger held-out evaluation set, more negative examples, more sentence-level examples, and more variation across child speakers, microphones, accents, and reading levels.

The local path works, but it is slower than ideal. A roughly 3-second reading attempt can take about 10 seconds end-to-end in the deployed Space. That is acceptable for a hackathon demo, but a real reading product needs tighter feedback.

The curriculum is intentionally tiny. That was the right hackathon tradeoff, but the next version needs a larger decodable sentence bank, parent controls, and a way to review attempts over time.

## What I Learned

Small models work best when the job is sharply defined.

Read-Along AI does not ask a language model to be a teacher, therapist, curriculum designer, and ASR system all at once. It asks each component to do one bounded thing:

- ASR turns audio into text.
- Exact matching handles obvious successes.
- MiniCPM judges close text variants.
- Local audio assets provide word help.
- The UI keeps the interaction simple and encouraging.

That division of labor is what made the app feel plausible under the Build Small constraint.

I also learned that privacy and UX are connected. Off the Grid Mode is not just a badge strategy. For a child-facing voice app, local inference changes the trust story. It lets the product say: this practice can happen in the room, with the family, without making a child's voice the price of feedback.

## What I Would Build Next

The next version would focus on:

- Real-time streaming ASR.
- Live word highlighting while the child reads.
- A larger child-speech evaluation set.
- More sentence-level training examples for the MiniCPM judge.
- Parent and teacher review tools.
- Better local latency.
- A richer curriculum organized by reading skill.

The long-term product idea is not to replace an adult. It is to make daily reading practice easier to start, easier to repeat, and gentler when a child is still finding the words.

## Links

- Space: https://huggingface.co/spaces/build-small-hackathon/read-along-ai
- Repository: https://github.com/kingkw1/read-along-ai
- Fine-tuned MiniCPM evaluator: https://huggingface.co/kingkw1/minicpm-phonetic-evaluator
- Hackathon social post: https://huggingface.co/posts/kingkw1/522163043386016
