# OpenAI assistants in Polymath

Polymath uses the OpenAI Responses API for four application workloads:

1. Administrator Chat Boss
2. Signed-in customer support
3. Paid virtual-teacher conversation and explicit camera snapshots
4. Create Music song-blueprint generation

All calls originate in the Node server. The browser never receives the OpenAI API key.

## Why two model tiers

- `OPENAI_CHAT_MODEL` defaults to `gpt-5.6-terra` for a strong cost/latency balance in frequent conversations.
- `OPENAI_MUSIC_MODEL` defaults to `gpt-6-astra` for the harder, less frequent songwriting blueprint task.
- `OPENAI_VISION_MODEL` defaults to the chat tier and runs only after the learner explicitly shares one snapshot.

Every model name is configuration, not hard-wired product logic. Change the environment variable, deploy a new task revision, run evaluations, and roll back if quality drops.

## Fine-tuning boundary

The flagship GPT-6 Astra and GPT-5.6 models are not fine-tunable. The optional supervised candidate therefore uses `gpt-4.1-mini-2025-04-14`, which is non-reasoning and supports supervised fine-tuning. `server/openAiResponses.js` detects GPT-4.1 and omits the unsupported `reasoning` field automatically.

OpenAI's current documentation says the fine-tuning platform is winding down and is unavailable to new fine-tuning users. A training job can run only if this OpenAI project already has access. The production models remain the prompt-engineered OpenAI models unless a candidate exists and passes the holdout gate.

Polymath's project returned HTTP 403 when tested on 2026-09-09, so no OpenAI
weight update occurred. The current production candidate instead shares a
versioned behavior contract across Chat Boss, support, and teacher workloads and
must pass repeated holdout evaluations. This is behavior/prompt tuning, not model
weight fine-tuning.

The optimization order is deliberate:

1. Establish deterministic evals first.
2. Keep training and validation prompts separate.
3. Validate JSONL, roles, uniqueness, and secret scanning locally.
4. Upload only the reviewed training and validation files.
5. Start an SFT job with automatic hyperparameters.
6. Compare the returned `ft:` model against the current baseline.
7. Promote only at 90% or better and only when it beats the baseline.

Local commands:

```powershell
cd C:\Users\admin\polymath_repo\server
npm run openai:finetune:validate
npm run openai:finetune:submit
node fine-tuning\cli.js status ftjob-REPLACE_ME
npm run openai:eval -- gpt-5.6-terra ft:gpt-4.1-mini-2025-04-14:REPLACE_ME
```

Fine-tuning run records and complete eval replies remain under `.local-dev/openai-fine-tuning`, which Git ignores. Generated upload files are also ignored. The source examples are versioned so every behavioral change can be reviewed.

## Request styles

Support and the older teacher endpoint are synchronous and use `store=false`. Chat Boss, Create Music, and paid virtual-teacher replies use background Responses jobs so they survive slow replies and page reloads.

The adapter is `server/openAiResponses.js`. It converts the existing Polymath job contract into OpenAI's response contract:

```text
queued       -> IN_QUEUE
in_progress  -> IN_PROGRESS
completed    -> COMPLETED
failed       -> FAILED
incomplete   -> FAILED
cancelled    -> CANCELLED
```

It also translates `max_tokens` to `max_output_tokens`, uses the configured reasoning effort, extracts text from Responses output items, validates response IDs, applies server timeouts, and returns bounded errors without exposing credentials.

## Production secret

The real `OPENAI_API_KEY` belongs in the existing `polymath/api-runtime` AWS Secrets Manager JSON object. ECS loads that object before importing `server.js`. Non-secret model names are injected by the deployment workflow.

Do not add the key to GitHub variables, workflow YAML, frontend environment variables, logs, or source files.

## What remains on RunPod

MuScriptor/Polymath audio transcription and ML evaluation still require GPU workers and network-volume weights. They continue to use `RUNPOD_SERVERLESS_ENDPOINT_ID`, `RUNPOD_API_KEY`, and the MuScriptor storage settings.

The owner's personal DeepSeek environment is not a Polymath dependency and this code does not access or mutate it.
