# Polymath Musician Architecture

## Frontend

React/Vite single-page application with route state managed in `src/App.jsx`.

- `pages/StudioPage` flow inside `App.jsx`: grand-piano playback, pedal-aware scheduling, and falling notes.
- `pages/GuitarPage.jsx`: guitar tablature/chord playback, fretboard interaction, timeline scrubbing, and ready-to-play imports.
- `pages/EnsemblePage.jsx`: visual teaching, playback, slowing, scrubbing, and ready-to-play lessons for fiddle, banjo, mandolin, dobro, upright bass, ukulele, electric guitar, drum set, and synth keyboard.
- `components/InstrumentTeacherSurface.jsx`: responsive physical-instrument SVG teachers with live press/fret/slide/strike targets.
- `components/MusicUploadPanel.jsx`: three minimal entry points for JSON/MIDI upload, local PDF optical music recognition, and MuScriptor audio/video transcription.
- `components/PdfTranslationPanel.jsx`: allowance/Mcoin choice, progress polling, ETA display, refund/failure messaging, and output download.
- `components/MediaTranscriptionPanel.jsx`: multipart media upload, rights confirmation, native MuScriptor progress polling, protected result download, and direct studio loading.
- `pages/MarketplacePage.jsx`: minimalist Composers catalogue, seller publishing, purchases, permanent verified reviews, follower controls, profiles, and downloads.
- `pages/AccountPage.jsx` and `pages/PaymentPage.jsx`: wallet, Pro allowance, USD/Mcoin catalogue, and PayPal entry points.
- `pages/AdminDatabasePage.jsx`: sectioned administrator console for responsive device previews, promotion management, policy controls, user records, and forced password recovery.

## Audio and score layer

- `engine/audioEngine.js`: sampled grand-piano engine.
- `engine/guitarEngine.js`: acoustic-guitar string and body model.
- `engine/ensembleEngine.js`: playable synthesis profiles for the additional melodic instruments plus dedicated electronic drum synthesis.
- `engine/scheduler.js`: timing normalization and duration calculation.
- `utils/songParser.js`: JSON, CSV, MIDI, and MusicXML ready-to-play ingestion. PDFs are intentionally routed to the translation queue.

## Backend

`server/server.js` currently provides:

- account/session endpoints;
- USD PayPal products and webhook verification;
- Mcoin wallets and account-wide cash-out records with a 25% fee and manual payout review;
- Composers publishing, server-side file validation, exact 25% fee allocation, purchases, permanent reviews, follows, profiles, and protected downloads;
- YouTube search proxy;
- authenticated provider-free PDF music-reading jobs, monthly allowance accounting, Mcoin charging, duplicate-job protection, ETA extensions, strict ready-to-play validation, confidence/diagnostics, and automatic restoration after failure;
- a three-level local OMR ladder: exact embedded MusicXML, semantic SMuFL/vector PDF reading, then conservative high-resolution computer vision for scans;
- authenticated disk-streamed audio/video jobs, bundled FFmpeg preparation, sequential MuScriptor execution, instrument constraints, native chunk progress, restart recovery, source cleanup, and protected ready-to-play results;
- backend-enforced signup/spending policies, percentage-only promotion codes, reusable short Friend IDs, redemption limits, administrator password-reset audits, and hashed 30-day sessions.

## Current persistence

The bundled development backend uses an atomically replaced local JSON database and filesystem uploads. Passwords use salted scrypt hashes and new session records contain SHA-256 token hashes. This keeps setup simple but is not safe for horizontally scaled production traffic.

## Production target

Use a transactional relational database, durable job queue, private object storage, managed secrets, rate limiting, malware scanning, observability, backups, and independently scalable translation workers. Financial actions must execute in database transactions with idempotency keys.

MuScriptor runtime can be invoked locally through `server/muscriptor_worker.py` or streamed to a remote GPU service through `MUSCRIPTOR_REMOTE_URL`. The remote client consumes MuScriptor's SSE note/progress events and produces the same protected ready-to-play JSON result. Model execution is disabled by default and controlled by `MUSCRIPTOR_ENABLED`, `MUSCRIPTOR_MODEL`, and the local or remote worker settings. The published weights are CC BY-NC 4.0 and must not be enabled for commercial use without separate permission.
