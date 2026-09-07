# ChatBoss candidate v002 decision

Candidate v002 is **trained, preserved, and rejected for production**. It was
not merged into the base model, copied to the approved-adapter directory, or
connected to the live endpoint.

## Reproducible identity

- Base model: `Qwen/Qwen3.5-35B-A3B`
- Immutable base revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- Training method: completion-only QLoRA, 4-bit NF4 double quantization
- LoRA rank / alpha: 16 / 32
- Training examples: 114
- Validation examples: 34
- Epochs: 2
- Temporary training GPU: NVIDIA H100 PCIe, 80 GB
- Candidate location: `/runpod-volume/chatboss/candidates/v002/`

## Training result

| Metric | Result |
| --- | ---: |
| Training loss | 2.6242 |
| Validation loss | 2.3893 |
| Training runtime | 442.4 seconds |
| Frozen-base behavior score | 69.66% |
| v002 behavior score | 71.46% |
| Score change | +1.80 percentage points |
| Frozen-base safety pass | 87.50% |
| v002 safety pass | 93.75% |

The lower validation loss proves that the adapter learned the curriculum. It
does **not** prove production quality. The independent behavior suite showed a
small improvement, but v002 missed the 85% quality gate and the 100% safety
gate.

## What improved

- OTP guidance became concise and refused secret sharing.
- Answers were generally shorter and more Polymath-specific.
- The teacher stopped inventing measurements when no practice report existed.
- Pedal feedback and frustration coaching improved.

## Why it failed

- It asserted that RunPod was not down without live infrastructure evidence.
- It did not always name all measured notes or exact values in one-step advice.
- It claimed the learner's wrists were too low while the camera was disabled.
- Its dynamics answer omitted the measured averages and regressed by 28.57
  percentage points on that case.
- Some product answers still treated “Musician” as a generic outside product.

## Decision

Keep v002 as experiment evidence only. Build v003 from more reviewed,
contrastive examples, expand the holdout suite, and require all automated gates
plus human review. Server-side deterministic guardrails must remain in place
even after a future adapter passes.
