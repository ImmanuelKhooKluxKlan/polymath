# Polymath Musician

Polymath Musician is a responsive browser music platform for learning, playing, translating, buying, and selling digital music sheets. It includes specialised piano and acoustic-guitar studios plus a visual Instrument Teacher for five-string fiddle, five-string banjo, mandolin, dobro/resonator guitar, upright double bass, ukulele, electric guitar, drum set, and synth keyboard.


## Visual teaching system

- The physical playing surface is the primary lesson view; the timing lane is secondary.
- Live colour targets show the exact key, string, fret/finger position, dobro slide position, bass position, or drum/cymbal to use.
- Every visual target is clickable for practice and remains usable with touch, mouse, or keyboard controls.
- Drum and synth modes support electronic-pop lessons while the bluegrass and acoustic instruments retain instrument-specific guidance.
- The piano sound preset is shown to users as **Polymath Musician render** and its built-in selector is labelled **Free available to play songs**.

## User-facing product model

- **Upload Ready-to-Play Sheet:** presents a minimal JSON/MIDI picker the platform can read immediately.
- **Translate to a Ready-to-Play Sheet:** accepts a readable instrumental PDF, shows an initial estimated time of about 20 minutes, and extends the estimate in five-minute blocks when necessary.
- **Transcribe Music Audio/Video:** signed-in users can upload MP3/audio or a music-video file for MuScriptor to turn into a ready-to-play JSON sheet. Piano and guitar use instrument constraints; Band automatically detects a multi-instrument mix.
- **Free accounts:** 1 included PDF translation per month.
- **Pro accounts:** 20 included PDF translations per month.
- **Mcoin alternative:** every account can pay 30 Mcoins for a translation instead of using its allowance.
- **USD peg:** $1 USD = 10 Mcoins, so one 30-Mcoin translation is the $3 equivalent.
- **Marketplace:** listings visibly identify PDF versus ready-to-play formats. Polymath Musician retains a 10% marketplace fee and credits 90% to the seller.

## Responsive support

The interface is designed for desktop, laptop, iPad/tablet, and phone layouts. Navigation, upload controls, marketplace cards, studio timelines, and payment choices collapse into touch-friendly layouts at smaller widths.

## Local setup

Requirements: Node.js 20 or newer and npm.

```bash
npm install
npm --prefix server install
cp .env.example .env
cp server/.env.example server/.env
npm run server
```

In a second terminal:

```bash
npm run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`.

## Environment configuration

The frontend only needs `VITE_API_BASE_URL` in the root `.env`.

The backend uses `server/.env` for PayPal, YouTube, and direct OpenAI PDF translation. Create an active **USD 19.99/month** PayPal plan for `PAYPAL_PRO_PLAN_ID`. PDF translation remains safely unavailable, with no user charge, until `OPENAI_API_KEY` is configured.

Audio/video transcription uses the Python `muscriptor` package and the selected local checkpoint. The server streams uploads to disk, uses its bundled FFmpeg binary to prepare the first ten minutes as mono audio, runs one MuScriptor job at a time, and exposes model progress to the browser. Set `MUSCRIPTOR_PYTHON`, select `MUSCRIPTOR_MODEL=large`, and set `MUSCRIPTOR_ENABLED=true` only after confirming your use is permitted. MuScriptor's published model weights are **CC BY-NC 4.0 (non-commercial)**; this switch must remain off for commercial use unless you obtain separate permission from the rights holders.

MuScriptor Large strongly benefits from a GPU. On the tested CPU-only Windows machine, a five-second sample required about 143 seconds of model generation. `MUSCRIPTOR_TIMEOUT_MINUTES` defaults to 360, and the sequential queue prevents concurrent Large-model loads; use a CUDA worker for practical full-song throughput.

PayPal is the only checkout provider. Add both `PAYPAL_CLIENT_ID` and `PAYPAL_SECRET_KEY` for one-time Mcoin checkout. Pro subscriptions also require `PAYPAL_PRO_PLAN_ID`; live webhook updates require `PAYPAL_WEBHOOK_ID`.

For local admin access, the provided `server/.env` authorizes `admin@polymath.local`. Register that account, then use **Admin sign in**. Replace `ADMIN_EMAILS` with your real administrator email before deployment.

Never commit either real `.env` file.

## Validation commands

```bash
npm run lint
npm run build
node --check server/server.js
npm --prefix server test
```

## Administrator console

Backend-authorized administrators sign in through **Account > Admin sign in** and use a focused console with separate Overview, Device Preview, Vouchers & Coupons, Rules & Policies, and Users & Passwords workspaces.

- Device Preview covers small phones, modern and large phones, foldables, tablets, iPads, laptops, desktops, large desktops, landscape rotation, and custom CSS viewport sizes.
- Mcoin vouchers credit a user's wallet from Account. Marketplace percentage and fixed-Mcoin coupons are applied by the backend during a purchase and support minimum spend, account-age, expiry, total-use, and per-user limits.
- Every account receives a stable public Friend ID in the form `user_aa123`: the `user_` prefix plus exactly five easy-to-type hexadecimal characters. Internal relational IDs remain private and unchanged.
- A Friend ID percentage voucher lets a signed-in buyer enter another registered user's Friend ID at marketplace checkout. The same Friend ID may be shared with any number of buyers; self-referrals are blocked, and entering an ID never signs in as or exposes the friend's account.
- Rules control registration availability, minimum signup age, minimum password length, minimum marketplace price, minimum withdrawal, new-user Mcoins, and public policy links/notices.
- Administrator password resets revoke every existing user session, issue a temporary password, and force the user to choose a private password at next sign-in.

## MuScriptor transcription flow

- Piano, guitar, and every visual instrument studio expose **Transcribe Music Audio/Video (MuScriptor)** beside the JSON/MIDI and PDF choices.
- Band exposes a blue expandable MuScriptor control under its general sheet; a completed full-mix transcription becomes the band's shared arrangement.
- Accepted input includes MP3, WAV, FLAC, OGG, M4A, AAC, MP4, MOV, WebM, MKV, AVI, MPEG, and MPG up to 100 MB.
- Users must sign in and confirm they have permission to transcribe the recording.
- Raw media and prepared WAV files are deleted after success or failure. The protected ready-to-play JSON result remains available to its owner.

## Local data persistence

The development backend persists registrations, users, salted scrypt password hashes, purchases, Mcoin ledger entries, policies, promotions, redemption records, password-reset audit events, and login events in `server/data/database.json`. Writes use a temporary file followed by a rename so an interrupted write is less likely to leave a partial database.

Browser bearer tokens are returned only to the signed-in client. The database stores a SHA-256 hash of each new session token with its user ID and 30-day expiry rather than storing the raw token. This JSON database is suitable for one local server process; production should use the transactional database and managed session storage described below.

## Production boundary

The delivered UI and application flows are release-oriented, but a high-scale public launch still requires production infrastructure: a transactional database, durable job queue, object storage, malware scanning, rate limiting, monitoring, backups, and legal/licensing review. See `PRODUCTION_CHECKLIST.md` and `MIGRATION_NOTES.md`.
