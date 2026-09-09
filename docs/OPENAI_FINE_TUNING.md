# Polymath OpenAI fine-tuning runbook

## Goal

Create a cheaper specialized assistant candidate without lowering teaching accuracy, evidence honesty, support security, or conversational quality.

The candidate is for Chat Boss, support, and virtual-teacher text. Create Music stays on the stronger flagship model until Polymath has a large collection of separately reviewed song blueprints. Mixing unreviewed generated lyrics into the assistant dataset would teach formulaic output and make both products worse.

## Dataset

The curated source is:

```text
server/fine-tuning/polymath-assistant.examples.json
```

It currently contains:

- 26 training examples
- 10 isolated validation examples
- music theory and instrument technique
- measured-performance grounding
- explicit missing-evidence responses
- account and credential boundaries
- adult-only companion tone with anti-dependency boundaries
- technical uncertainty and architecture explanations

`buildDataset.js` converts the source into OpenAI chat-format JSONL. `openAiFineTuning.js` rejects fewer than ten examples, duplicate user prompts, bad roles, missing assistant answers, malformed JSON, and strings that resemble API or private keys.

## Evaluation gate

`assistant-evals.json` is separate from both training and validation data. It checks required facts, forbidden claims, and response length. A candidate is not production-ready merely because training loss fell.

A release candidate must:

- pass at least 90% of deterministic eval cases;
- beat or tie the currently deployed baseline;
- make no invented hearing, vision, payment, or account-action claims;
- keep credential-protection cases at 100%;
- pass a short human tone review;
- survive the full Polymath backend test suite.

## Commands

Validate without spending money:

```powershell
cd C:\Users\admin\polymath_repo\server
npm run openai:finetune:validate
```

Submit a paid training job after `OPENAI_API_KEY` is configured and project access is verified:

```powershell
npm run openai:finetune:submit
```

If a submission is rejected after files were uploaded, the CLI now removes both
temporary files automatically. To remove Polymath-named fine-tuning uploads from
an earlier interrupted attempt:

```powershell
npm run openai:finetune:cleanup
```

The command prints a non-secret job ID and writes a run record. Check it later:

```powershell
node fine-tuning\cli.js status ftjob-REPLACE_ME
```

Compare baseline and candidate:

```powershell
npm run openai:eval -- gpt-5.6-terra ft:gpt-4.1-mini-2025-04-14:REPLACE_ME
```

Do not promote a model when the evaluator exits with code 2. That means at least one tested model is below the 90% gate.

## Promotion and rollback

The model ID is configuration, never source-code logic.

Promotion:

1. Set GitHub repository variable `OPENAI_CHAT_MODEL` to the successful `ft:` ID.
2. Deploy a new immutable ECS task revision.
3. Run support and virtual-lesson smoke tests.
4. Observe error rate, latency, token usage, and safety rejections.

Rollback:

1. Restore `OPENAI_CHAT_MODEL=gpt-5.6-terra`.
2. Redeploy the previous known-good application revision.
3. Keep the failed candidate and eval report for diagnosis; do not silently retry it with production traffic.

## Access limitation

As of 2026-09-09, official OpenAI documentation says organizations without prior
fine-tuning history cannot create training jobs. Polymath's project was tested on
that date and OpenAI returned HTTP 403 before creating a job. No OpenAI weights
were changed.

Polymath therefore uses the supported fallback: the versioned server-owned
behavior contract in `server/assistantBehavior.js`, deterministic application
tools, retrieval, and holdout evaluation. After correcting false-negative rubric
rules, the untuned baseline was 9/12 (75%). The behavior-tuned GPT-5.6 Terra
configuration then passed three independent runs, 36/36 checks (100%). These are
small contract tests, not a claim of universal model accuracy. Keep collecting
real, consented, de-identified failures and expand the holdout set before making
broad quality claims.

Do not claim that weight fine-tuning occurred when the API did not create a job.
