# Polymath ChatBoss training lane

This folder improves the **conversation model**, not the audio-to-MIDI model.

## What is connected now

`Qwen/Qwen3.5-35B-A3B` is the protected base checkpoint behind the RunPod
Serverless endpoint named `polymath-chat-boss`. The Node API sends it one of two
separate, grounded roles:

1. **Customer service** — explains the product, but cannot pretend to change an
   account, payment, password, refund, or balance.
2. **Piano teacher** — coaches from measured lesson evidence and must not invent
   a note, duration, camera event, or audio event.

The base checkpoint stays unchanged. Training creates a small, versioned LoRA
adapter. That makes rollback possible.

## Storage layout

```text
/runpod-volume/chatboss/
├── huggingface-cache/        downloaded base checkpoint cache
├── datasets/
│   └── v001/                 reviewed train and validation JSONL
├── candidates/
│   └── v001/                 unapproved adapter and trainer state
├── adapters/
│   └── v001/                 approved adapter only
├── merged/
│   └── v001/                 optional full merged checkpoint for vLLM
└── evals/
    └── v001/                 base/candidate comparison reports
```

Think of the base checkpoint as a capable graduate. The LoRA is a small Polymath
staff handbook clipped onto it. We change the handbook first, never erase the
graduate's original education.

## Safe workflow

### 1. Add reviewed examples

Add original examples to `data/seed_examples.jsonl`. Never paste customer chats
directly into training. Remove names, email addresses, phone numbers, payment
identifiers, secrets, and anything a user did not consent to train on.

Each example ends with the desired assistant answer and includes a `role` value
of either `support` or `teacher`.

### 2. Validate and split

```bash
python ml/chatboss/build_dataset.py \
  --input ml/chatboss/data/seed_examples.jsonl \
  --output /runpod-volume/chatboss/datasets/v001
```

This rejects malformed conversations and likely secrets/PII, removes exact
duplicates, and creates deterministic train/validation splits. It does not make
weak examples good; a human still reviews every desired answer.

Upload an approved split to the existing ChatBoss network volume with:

```bash
node ml/chatboss/upload_dataset.mjs \
  --env /private/path/to/server.env \
  --chatboss-env /private/path/to/chatboss.env \
  --root ml/chatboss/prepared/v001 \
  --version v001
```

The ChatBoss volume is in `US-KS-2`, so its private ChatBoss environment file
must use `RUNPOD_CHAT_BOSS_S3_REGION=US-KS-2` and
`RUNPOD_CHAT_BOSS_S3_ENDPOINT=https://s3api-us-ks-2.runpod.io/`. S3 endpoints
are tied to the volume's data centre; the main transcription volume can use a
different endpoint.

### 3. Train a candidate adapter on a separate GPU Pod

Do **not** train inside the vLLM Serverless inference worker. It is an inference
service, not a training environment.

```bash
python ml/chatboss/train_lora.py \
  --train /runpod-volume/chatboss/datasets/v001/train.jsonl \
  --validation /runpod-volume/chatboss/datasets/v001/validation.jsonl \
  --output /runpod-volume/chatboss/candidates/v001
```

The default safety gate requires at least 100 reviewed training examples. Use
hundreds of varied examples before judging quality. The script uses 4-bit NF4
QLoRA, rank 16, alpha 32, completion-only loss, gradient checkpointing, and a
validation set. Only adapter parameters update; the base weights remain frozen.

`create_candidate_v001.py` supplies an original bootstrap curriculum covering
support policies and evidence-grounded piano coaching. It contains no exported
customer conversations. Its output is still a **candidate**, not automatic
permission to replace the live model.

`run_candidate.py` executes `verify_chat_boundaries.py` before training. This
checks that the tokenizer places the completion at the exact boundary expected
by completion-only loss. A mismatch blocks the run before optimizer steps begin.

### 4. Evaluate before deployment

Compare the base model and the candidate on a holdout set that was not used for
training. Minimum gates should cover:

- support accuracy and correct escalation;
- no invented account actions;
- no requests for passwords, OTPs, keys, or card details;
- piano advice grounded in supplied observations;
- useful one-step corrections;
- no regression on normal conversation.

Loss alone is not approval. A lower validation loss can still produce a worse or
less safe assistant.

### 5. Promote, then serve

Copy an approved adapter from `candidates/v001` to `adapters/v001`, or merge it
with the base checkpoint into `merged/v001`. RunPod vLLM can use a local model
path through `MODEL_NAME`; its managed LoRA list is intended for adapters hosted
as Hugging Face repositories. Keep the old Serverless revision available for
instant rollback.

## Important boundary

The application does **not** silently learn from live chats. Production
conversations are not training data. A future admin review/export tool must use
explicit consent, redaction, review, and version approval before an example can
enter this lane.

## Recorded experiments

- [`CANDIDATE_V002.md`](./CANDIDATE_V002.md) records the first full QLoRA run.
  It improved the behavior score but failed the production gates, so it remains
  quarantined on the network volume.
- [`CANDIDATE_V003.md`](./CANDIDATE_V003.md) records the expanded-curriculum
  run. It regressed below the frozen base and remains quarantined. Its failure
  led to an important architecture correction: exact practice measurements are
  now preserved and rendered by deterministic coaching logic instead of asking
  a language model to recreate missing facts.
- `download_evidence.mjs` exports immutable JSON evidence from the private
  RunPod volume without downloading or exposing the model weights.
