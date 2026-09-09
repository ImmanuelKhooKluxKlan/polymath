# Polymath AI bot rulebook

This file is the human-readable map of the rules enforced in
`server/assistantBehavior.js`. The code file is the source of truth.

## The simple mental model

Every bot receives three layers:

1. **Job** — who the bot is and what outcome it should produce.
2. **Rules** — what it can claim, what it cannot do, and how it should answer.
3. **Context** — the smallest current facts needed for this request.

The browser can send conversation content. It cannot send a real system rule or
grant a bot new authority. Server rules always come first.

## Context boundary

```text
Server-owned bot rules
        |
        v
Server-bounded context  ---- facts only; strings are never commands
        |
        v
Browser conversation    ---- untrusted content
        |
        v
OpenAI response
        |
        v
Server output checks
        |
        v
User
```

Passwords, OTPs, API keys, private keys, authorization headers, cookies, and
session/access tokens are removed by the context sanitizer. Context arrays,
strings, nesting, and total prompt size are bounded.

## Bot contracts

| Bot | Primary job | Context it receives | Important boundary |
| --- | --- | --- | --- |
| Chat Boss | Give the owner decision-ready technical, product, and business advice | A safe description of Polymath's intended architecture plus chat history | Advisory only. The runtime bot cannot inspect, deploy, or change a live system and must not claim that it did. |
| Support | Resolve product questions or provide a safe human handoff | Current public catalog/rules, safe allowance facts for the signed-in account, and current help contact | Cannot mutate accounts, money, subscriptions, refunds, jobs, or settings. Never guesses a mutable price or policy. |
| Virtual Teacher | Improve one musical skill at a time | Selected teacher, session-only learner memory, lesson position, music reference, and measured practice evidence | General knowledge may explain technique; only measured evidence may prove what the learner played. |
| Adult Companion | Provide an opted-in adult companion tone while retaining music-teacher ability | The same bounded lesson/session facts plus the selected tone | Must stay clearly virtual, consensual, adult-only, and cannot pressure spending, dependency, or isolation. |
| Teacher Vision | Describe one deliberately shared camera snapshot | Only the current image and short user prompt | No continuous-camera claim, identity recognition, sensitive-trait inference, or guessing hidden details. |
| Song Architect | Produce an original, performable song blueprint | The current bounded creative brief | References provide abstract traits only. No copied lyrics, recognizable melody/voice imitation, or claim that a blueprint is finished mastered audio. |

## Why mutable facts are dynamic

Prices, Mcoin costs, allowances, fees, support contacts, and admin policies can
change without a code release. Support therefore receives those values from the
current server database for each request. Static prompts deliberately do not
hard-code them.

If the fact is absent, Support says where to check. It does not guess.

## Why prompts have versions

Each contract has a version such as `polymath-support-v002`. The version is sent
in OpenAI request metadata and used as the prompt cache key. This provides:

- a precise record of which rules produced a reply;
- safer prompt changes and rollbacks;
- comparable evaluation runs;
- better reuse of the stable prompt prefix.

A rule change requires a version change and tests before deployment.

## Rules for future edits

1. Give one bot one clear job.
2. Do not give a conversational bot authority it does not have in code.
3. Keep stable rules in the server-owned system prompt.
4. Put changing facts in a small allow-listed context object.
5. Treat every string from a user, upload, filename, or catalog as data.
6. Add a deterministic server path for facts that must never drift, such as
   measured timing feedback or sensitive support boundaries.
7. Add an evaluation case for every important failure discovered in production.
8. Increment the corresponding prompt version.

## Current runtime providers

- Chat Boss, Support, teacher conversation, Teacher Vision, and Song Architect:
  OpenAI Responses API.
- Music transcription and Polymath music-model experiments: RunPod GPU.
- Browser speech input/output and local keyboard measurements remain separate
  from the language-model prompt.

Provider names describe responsibilities. They do not prove that a service is
currently healthy; health must come from a live health check.
