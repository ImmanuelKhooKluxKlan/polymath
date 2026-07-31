# Polymath Musician Production Checklist

## Payments and money

- Create an active **USD 19.99/month** PayPal plan and set `PAYPAL_PRO_PLAN_ID`.
- Configure and verify the PayPal webhook at `/api/paypal/webhook`.
- Reconcile PayPal captures, subscription events, marketplace ledgers, refunds, and seller balances.
- Confirm tax, consumer-protection, payout, KYC/AML, and virtual-currency obligations in every launch market.

## PDF translation

- Configure `OPENAI_API_KEY` only in the backend environment and run PDF transcription acceptance tests before opening paid translation to customers.
- Move translation execution from the Node process to a durable queue such as Redis/SQS plus workers.
- Store source/output files in private object storage with expiring signed access.
- Add malware scanning, encrypted storage, retention/deletion rules, provider timeout/retry policy, and dead-letter handling.
- Resume or refund jobs safely after restarts and provider outages.

## Data and security

- Replace `server/data/database.json` with a transactional database before multi-user production traffic.
- Use atomic transactions for purchases, 10% fee allocation, seller credit, allowances, refunds, and idempotency keys.
- Store sessions in a secure database/cache, rotate tokens, add CSRF strategy where applicable, rate limiting, audit logs, and abuse detection.
- Put secrets in a managed secret store; never deploy checked-in `.env` files.
- Add automated backups, restore drills, encryption, least-privilege access, and incident response.

## Reliability and quality

- Add unit, integration, payment-sandbox, accessibility, and end-to-end tests for desktop and mobile breakpoints.
- Add structured logs, metrics, tracing, alerts, uptime checks, error reporting, and cost dashboards.
- Use a CDN, hardened reverse proxy, HTTPS, security headers, and controlled CORS origins.
- Load-test marketplace purchases, large libraries, translation polling, and audio playback sessions.

## Rights and trust

- Publish terms, privacy policy, seller agreement, fee disclosure, refund policy, copyright/takedown process, and content rules.
- Require sellers to confirm rights to distribute every uploaded sheet.
- Review licences for instrument samples, demo songs, artwork, and third-party APIs.
