# Polymath scaling and deployment runbook

Status: code prepared; production traffic has **not** been switched.

## The target in one picture

```text
Music listener (America / Asia-Pacific)
                 |
                 v
      Cloudflare DNS + TLS + cache
        |          |             |
        |          |             +--> R2 private artifacts
        |          |                  uploads, purchases, AI results
        |          |
        |          +--> R2 public instrument samples
        |               piano, guitar, strings, drums...
        |
        +--> Pages: React app (nearest edge)
        |
        +--> /api geographic load balancer
                 |                 |
                 v                 v
          AWS Ohio API       AWS Singapore API
                 \                 /
                  +--> PostgreSQL state
                  +--> SQS job queue --> RunPod GPU
```

Simple analogy: Cloudflare is the receptionist, the two APIs are two branch
offices, PostgreSQL is the shared account book, R2 is the shared filing room,
SQS is the numbered job tray, and RunPod is the rented specialist workshop.

## Why each move exists

| Old bottleneck | New owner | Result |
|---|---|---|
| 296.5 MiB instruments sent by one server | Cloudflare R2/cache | Samples download near the listener and are cached for a year. |
| React files sent by Ohio | Cloudflare Pages | Static UI is globally cached and does not consume API CPU. |
| One JSON database file | PostgreSQL-compatible state store | Concurrent regional writes are conflict checked. |
| Uploads kept on one disk | Private S3/R2 artifact store | Either region can process or download the same file. |
| In-process AI queue | SQS | Jobs survive restarts and can be consumed by more workers. |
| Mutable deployment | SHA-tagged containers | Every release is traceable and rollback is deterministic. |

## Safety limits

- Cloud deployment workflows are OFF until their matching `*_ENABLED` GitHub
  variable is exactly `true`.
- RunPod remains scale-to-zero with zero active workers. Set a small maximum
  worker count first; AI usage, not ordinary website visits, wakes workers.
- Instrument objects use versioned paths (`v1/...`) and immutable caching.
- Private artifacts never use a public R2 custom domain.
- Database balance conflicts return HTTP 409 instead of silently losing Mcoins.
- AWS budget alerts should be created at USD 200, 400 and 650. A budget is an
  alarm, not a guaranteed hard spending cap.

## Activation order (do not skip)

1. Keep the existing Ohio Lightsail site live.
2. Create the public instrument R2 bucket and private artifact bucket.
3. Run the instrument workflow and verify a piano and guitar sample through R2.
4. Deploy the frontend to Pages on a temporary hostname.
5. Back up `database.json` and `uploads`, and verify archive checksums.
   Preview the non-destructive legacy upload with
   `npm --prefix server run artifacts:sync-legacy -- --dry-run`, then run it
   without `--dry-run` after the private bucket credentials are configured.
6. Create PostgreSQL and import the existing JSON state by starting one API only.
7. Create SQS + dead-letter queue, then enable shared private artifacts.
8. Deploy the Ohio container and perform signup, OTP, Mcoin, PayPal webhook,
   marketplace download, PDF and audio translation smoke tests.
9. Add Singapore, repeat smoke tests, then add geographic API routing.
10. Move the public domain only after both origins pass health checks.

Rollback is the reverse traffic change: point Cloudflare back to Lightsail.
Never delete the old server, JSON backup, or upload backup during validation.

## GitHub configuration

Repository variables (not secrets):

```text
CLOUDFLARE_PAGES_ENABLED=false
R2_DEPLOY_ENABLED=false
AWS_ECS_DEPLOY_ENABLED=false
RUNPOD_IMAGE_BUILD_ENABLED=false
VITE_API_BASE_URL=https://api.polymathmusician67.com
VITE_ASSET_BASE_URL=https://audio.polymathmusician67.com
VITE_ASSET_RELEASE=v1
CLOUDFLARE_PAGES_PROJECT=polymath-musician
INSTRUMENT_R2_BUCKET=polymath-instruments
INSTRUMENT_ASSET_RELEASE=v1
AWS_ECR_REPOSITORY=polymath-api
```

Repository secrets:

```text
CLOUDFLARE_API_TOKEN
CLOUDFLARE_ACCOUNT_ID
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
AWS_GITHUB_DEPLOY_ROLE_ARN
```

Application runtime secrets belong in AWS Secrets Manager/ECS or the existing
server `.env`, never GitHub history. These include PayPal, OTP mail, OpenAI,
RunPod, admin-password bootstrap and database credentials.

## Cost posture under USD 800/month

Normal static listening should be cheap because Pages and R2 do the work.
The expensive variable is AI GPU time. Start with one small API task per region,
one small single-AZ PostgreSQL instance, two load balancers, zero active RunPod
workers and a maximum of three flex workers. At prototype traffic this should be
well below the ceiling; 24/7 GPU workers can consume the ceiling by themselves.

Recalculate with live metrics before raising any maximum. Track:

- RunPod billed seconds per completed translation;
- API requests, CPU, memory and p95 latency by region;
- R2 Class A/B operations and stored GiB;
- PostgreSQL CPU, connections and storage;
- cross-region database traffic and SQS retries;
- dead-letter queue depth.

## Current database limitation

The PostgreSQL adapter is a safe migration bridge: it stores the existing app
state as one conflict-checked JSONB document. This is suitable for moving the
current tiny production database without rewriting 70 routes in one dangerous
release. It is **not** the final schema for hundreds of thousands of registered
accounts. Normalize users, sessions, subscriptions, ledger, bands, listings and
jobs into indexed tables before account data becomes large.

Thirty thousand daily visitors are not the same as thirty thousand concurrent
users. Scale from measured concurrent requests and p95 latency, not visitor count.
