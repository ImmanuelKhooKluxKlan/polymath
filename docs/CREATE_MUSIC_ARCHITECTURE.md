# Polymath Create Music

## Product boundary

Create Music is a human-led songwriting and rehearsal studio. It does not render a finished fake music video. The user directs the song, edits every lyric, chooses the tempo and instruments, and performs the vocal.

## Runtime flow

```text
Browser: Idea -> Sound choices -> Editable lyrics
                    |
                    | authenticated, entitled AI request
                    v
AWS Node API -> OpenAI Responses API -> validated song blueprint
                    |
                    v
Browser deterministic arranger
  -> chords + guide melody + bass + drums + textures
  -> karaoke lyric timing
  -> Piano / Guitar / Ensemble audio engines
  -> MIDI, JSON, and lyric exports
```

OpenAI proposes language and high-level musical direction. It does **not** emit an unchecked MIDI file. The server requires a strict JSON Schema response, validates and bounds every field, and then gives that blueprint to Polymath's deterministic arranger. The arranger converts it into note events, timing, dynamics, sections, lyric cues, and pedal events. This separation keeps output playable even when a language-model suggestion is imperfect.

## Model isolation

- Polymath calls OpenAI through the server-side Responses API.
- The default high-quality music model is set by `OPENAI_MUSIC_MODEL`.
- Polymath has no access to OpenAI's underlying model weights and cannot overwrite them.
- The owner's personal DeepSeek checkpoint, RunPod endpoint, and network volume are outside this feature and are not read, called, trained, or modified.
- MuScriptor transcription remains a separate RunPod workload.

## Durable jobs

Song drafts use OpenAI background responses. The browser receives a response ID immediately and polls the Polymath API. The Polymath API checks OpenAI by that ID. This means a browser refresh or a slower model response does not require one long HTTP connection.

```text
POST /api/music-creation/jobs
  -> POST /v1/responses with background=true
  -> save response ID and bounded brief

GET /api/music-creation/jobs/:id
  -> GET /v1/responses/:response_id
  -> queued | in_progress | completed | failed
  -> validate blueprint before returning it

POST /api/music-creation/jobs/:id/cancel
  -> POST /v1/responses/:response_id/cancel
```

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
- The API stores the bounded brief and validated blueprint, not API keys.
- Synchronous support requests use `store=false`.
- Background music and teacher jobs must remain retrievable while active and are referenced only by their opaque response IDs.

## Environment variables

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_ORGANIZATION=
OPENAI_PROJECT=
OPENAI_CHAT_MODEL=gpt-5.6-terra
OPENAI_MUSIC_MODEL=gpt-6-astra
OPENAI_VISION_MODEL=gpt-5.6-terra
OPENAI_CHAT_REASONING_EFFORT=low
OPENAI_MUSIC_REASONING_EFFORT=medium
OPENAI_VISION_REASONING_EFFORT=low
OPENAI_TIMEOUT_MS=50000
OPENAI_VISION_TIMEOUT_MS=120000
```

`OPENAI_API_KEY` remains server-side in AWS Secrets Manager. Never put it in a `VITE_` variable, browser code, Git, screenshots, or downloaded projects.

## Cost controls

- Chat/support uses the configurable balanced model and low reasoning effort.
- Create Music uses the configurable flagship model and medium reasoning effort because blueprint quality matters more there.
- Inputs, history lengths, output tokens, and generated arrays are bounded in code.
- Requests run only when a user invokes an AI feature; there is no always-on OpenAI worker charge.
- Admins can switch model IDs through GitHub environment variables without editing application code.
