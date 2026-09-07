# Polymath Create Music

## Product boundary

Create Music is a human-led songwriting and rehearsal studio. It does not render a finished fake music video. The user directs the song, edits every lyric, chooses the tempo and instruments, and performs the vocal.

## Runtime flow

```text
Browser: Idea -> Sound choices -> Editable lyrics
                    |
                    | authenticated, entitled AI request
                    v
AWS Node API -> RunPod Serverless -> DeepSeek V4 (read-only inference)
                    |
                    v
          strict validated song blueprint
                    |
                    v
Browser deterministic arranger
  -> chords + guide melody + bass + drums + textures
  -> karaoke lyric timing
  -> Piano / Guitar / Ensemble audio engines
  -> MIDI, JSON, and lyric exports
```

DeepSeek decides language and high-level musical direction. It does **not** emit an unchecked MIDI file. The deterministic arranger converts its validated JSON blueprint into bounded note events, timing, dynamics, sections, lyric cues, and pedal events. This separation keeps output playable even if a language-model response is imperfect.

## Checkpoint isolation

- `RUNPOD_POLYMATH_CREATE_*` is a separate application namespace.
- The existing DeepSeek V4 checkpoint is used for inference only.
- No endpoint in this feature writes weights, adapters, or configuration to the source volume.
- The code reports `originalCheckpointPolicy: read-only` so this rule is visible in diagnostics.

If Create Music is fine-tuned later, first create a separate paid network volume and a versioned path such as:

```text
/runpod-volume/models/polymath-create/
  parent-manifest.json
  adapters/v001/
  merged/v001/
  serving/v001-q4/
```

Train a LoRA adapter against a pinned BF16/FP16 parent checkpoint, evaluate it on held-out songwriting briefs, then quantize the approved merged artifact for a **new** serving endpoint. Never train against or overwrite the personal DeepSeek volume. The current Q4 MXFP4 file is a serving artifact, not the preferred training source.

## Access and billing

Administrators create categories and plans in **Admin -> Subscription catalog**. Customer-facing feature text and server-enforced entitlements are separate on purpose. A label cannot accidentally grant access.

Supported Create Music entitlements are:

- `create_music.projects`
- `create_music.ai_guidance`
- `create_music.arrangements`
- `create_music.exports`
- `create_music.guide_voice`

Paid plans require their matching PayPal plan ID before publication. Once a plan has a purchase record, its price, currency, billing interval, and PayPal plan ID are immutable. Archive it and create a new version instead.

## Storage and privacy

- Unsaved drafts autosave in that browser's local storage.
- Cloud projects require authentication and a matching entitlement.
- Every project query is scoped to the authenticated user ID.
- Optimistic revisions reject stale-tab overwrites with HTTP 409.
- AI jobs are owned by one user and cannot be polled or cancelled by another.
- The API stores the bounded brief and validated blueprint, not model secrets.

## Environment variables

```dotenv
RUNPOD_POLYMATH_CREATE_ENDPOINT_ID=
RUNPOD_POLYMATH_CREATE_MODEL=polymath-create-deepseek-v4
RUNPOD_POLYMATH_CREATE_DISPLAY_MODEL=DeepSeek V4 (read-only source)
RUNPOD_POLYMATH_CREATE_TIMEOUT_MS=1200000
```

`RUNPOD_API_KEY` remains server-side. Never put it in Vite variables, browser code, Git, or a downloaded project.

## Current performance observation

The 2026-09-07 live contract test waited about 424.8 seconds for the scaled-to-zero worker, then completed inference in about 17.5 seconds and returned a valid blueprint. The long first response is therefore a cold-start/capacity issue. Manual arranging and rehearsal do not depend on that worker and remain available while it is cold.
