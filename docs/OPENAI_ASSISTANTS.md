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
