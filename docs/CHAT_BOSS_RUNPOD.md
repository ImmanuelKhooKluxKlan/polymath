# Archived: former Polymath Chat Boss on RunPod

> Historical deployment record only. Polymath no longer calls this endpoint. Chat Boss, support, virtual-teacher conversation, and Create Music now use the OpenAI Responses API. Personal RunPod models are outside the Polymath project and must not be changed by application deployments.

This deployment is intentionally separate from the existing MuScriptor endpoints.

## Live resources

- Serverless endpoint: `polymath-chat-boss` (`lk3brapzdbx0ch`)
- Worker template: `polymath-chat-boss-qwen35-v1` (`u3k8kod5k4`)
- Persistent network volume: `Polymath Chat Boss Weights` (`ccppo4yp9l`)
- Region: `US-KS-2`
- Volume capacity: 250 GB
- Baseline checkpoint: `Qwen/Qwen3.5-35B-A3B`
- Pinned checkpoint revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- Served model alias: `polymath-chat-boss`

The endpoint has zero minimum workers and one maximum worker. It therefore incurs
no always-on GPU worker charge, but the persistent network volume remains billed
until it is deleted. The first request after scaling to zero has a cold start.

## API contract

Use the OpenAI-compatible base URL:

```text
https://api.runpod.ai/v2/lk3brapzdbx0ch/openai/v1
```

Authenticate with `RUNPOD_API_KEY` and send `polymath-chat-boss` as the model.
The server-side helper is `server/chatBossRunpod.js`. It disables Qwen's thinking
mode by default so internal reasoning is not emitted in user-facing answer text;
callers can override `chat_template_kwargs` when a different mode is required.

Allow up to 15 minutes at the application boundary for a scale-to-zero cold
start. The validated warm smoke test completed in under one second; cold-start
time depends mainly on GPU placement and reading the checkpoint from the volume.

## Weight lifecycle

The attached volume is mounted at `/runpod-volume`. The worker stores its
Hugging Face cache below `/runpod-volume/chatboss`, so downloaded model shards
survive worker shutdowns and replacement.

Keep releases immutable:

```text
/runpod-volume/chatboss/
  huggingface-cache/       # official baseline cache
  datasets/                # versioned, cleaned training data
  adapters/v001/           # LoRA/QLoRA checkpoints
  merged/v001/             # merged model releases
  evals/v001/              # evaluation results and decisions
```

Do not fine-tune the local `Qwen3.5-35B-A3B-Base` checkpoint as though it were a
chat model. The live baseline is the official post-trained checkpoint. Once the
training dataset and evaluation gates are ready, train an adapter from the
post-trained model, evaluate it, then point `MODEL_NAME` at the approved merged
or adapter-backed release.

## Current inference profile

- One 96 GB-or-larger GPU, preferring RTX PRO 6000 Blackwell Server Edition
- CUDA 13.0 worker image
- BF16 weights
- 65,536-token context during initial validation
- Maximum 32 scheduled sequences and two RunPod jobs per worker
- Safetensors prefetch from the persistent Hugging Face cache
- Text-only serving to preserve KV-cache capacity
- Reasoning and tool-call parsers disabled for the baseline smoke test; enable them
  only after the plain chat path has passed validation

The endpoint should be changed to a tested FP8 or AWQ release before sustained
production traffic if a 48 GB GPU cost target is desired.
