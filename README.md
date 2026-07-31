# Polymath Musician

Polymath Musician is a responsive browser music platform for learning, playing, translating, buying, and selling digital music sheets. It includes specialised piano and acoustic-guitar studios plus a visual Instrument Teacher for five-string fiddle, five-string banjo, mandolin, dobro/resonator guitar, upright double bass, ukulele, electric guitar, drum set, and synth keyboard.


## Visual teaching system

- The physical playing surface is the primary lesson view; the timing lane is secondary.
- Live colour targets show the exact key, string, fret/finger position, dobro slide position, bass position, or drum/cymbal to use.
- Every visual target is clickable for practice and remains usable with touch, mouse, or keyboard controls.
- Drum and synth modes support electronic-pop lessons while the bluegrass and acoustic instruments retain instrument-specific guidance.
- The piano sound preset is shown to users as **Polymath Musician render** and its built-in selector is labelled **Free available to play songs**.

## User-facing product model

- **Upload Ready-to-Play Sheet:** accepts JSON, MIDI, MusicXML, XML, and CSV formats the platform can read immediately.
- **Translate to a Ready-to-Play Sheet:** accepts a readable instrumental PDF, shows an initial estimated time of about 20 minutes, and extends the estimate in five-minute blocks when necessary.
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

Never commit either real `.env` file.

## Validation commands

```bash
npm run lint
npm run build
node --check server/server.js
```

## Production boundary

The delivered UI and application flows are release-oriented, but a high-scale public launch still requires production infrastructure: a transactional database, durable job queue, object storage, malware scanning, rate limiting, monitoring, backups, and legal/licensing review. See `PRODUCTION_CHECKLIST.md` and `MIGRATION_NOTES.md`.
