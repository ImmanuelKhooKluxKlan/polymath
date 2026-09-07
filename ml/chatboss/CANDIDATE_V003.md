# ChatBoss candidate v003 decision

Candidate v003 is **trained, preserved, and rejected for production**. It was
not merged, promoted, or connected to the live endpoint. The temporary H100 Pod
was deleted after the evidence was downloaded; the candidate remains
quarantined on the ChatBoss network volume.

## Reproducible identity

- Base model: `Qwen/Qwen3.5-35B-A3B`
- Immutable base revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- Training method: completion-only QLoRA, 4-bit NF4 double quantization
- LoRA rank / alpha: 16 / 32
- Training examples: 204
- Validation examples: 70
- Independent behavior holdout: 33 cases
- Epochs: 3
- Learning rate: `0.000075`
- Temporary training GPU: NVIDIA H100 PCIe, 80 GB
- Candidate location: `/runpod-volume/chatboss/candidates/v003/`
- Adapter SHA-256: `92ba2ecd8688acaa8a707b8affe3192a893f44c9d73127d46b33475a82f030e0`

All 274 train and validation examples passed the completion-token boundary
preflight before optimizer steps began.

## Training result

| Metric | Result |
| --- | ---: |
| Training loss | 2.3918 |
| Validation loss | 2.1404 |
| Validation token accuracy | 49.38% |
| Training runtime | 1,119.6 seconds |
| Frozen-base behavior score | 64.21% |
| v003 behavior score | 63.03% |
| Score change | -1.18 percentage points |
| Frozen-base safety pass | 93.94% |
| v003 safety pass | 93.94% |
| Frozen-base support score | 63.02% |
| v003 support score | 67.29% |
| Frozen-base teacher score | 65.33% |
| v003 teacher score | 59.02% |

These behavior scores are results from a fixed lexical and safety rubric. They
are not a claim of real-world accuracy. Lower validation loss showed that v003
learned its small curriculum, while the untouched holdout showed that the
learned behavior did not generalize well enough.

## Why it failed

- Every automated production gate failed.
- Candidate behavior fell below the frozen base overall.
- Teacher behavior regressed by 6.31 percentage points.
- Two safety cases failed: queue-status wording and cold-start/GPU-status
  wording.
- 11 cases scored below the minimum 60% floor.
- 32 of 33 cases remained below the 85% target.
- Exact evidence was often omitted: note names, millisecond timing, target and
  actual hold duration, dynamics averages, and stale-report identity.
- The training curriculum mostly taught generic correction patterns. It did not
  contain enough varied input-to-exact-output pairs that preserved changing
  note names and numeric measurements.

## Engineering correction

Do not solve factual grounding only by adding more epochs. The application now
preserves exact per-note timing, hold, dynamics, and pedal evidence. Measured
feedback is rendered deterministically from that evidence before ChatBoss is
called. ChatBoss remains useful for natural conversation, explanations, and
encouragement, but it is not trusted to recreate discarded measurements.

Sensitive support boundaries - credentials, billing actions, live service
status, current prices and limits, privacy, and training consent - are also
answered by deterministic product rules before model inference.

## Decision

Keep v003 as experiment evidence only. Do not deploy it and do not train v004
by blindly increasing epochs. A future candidate needs a larger reviewed
curriculum with varied exact values, a production-representative system prompt,
lower-risk hyperparameters, and the same independent safety gates plus human
review.
