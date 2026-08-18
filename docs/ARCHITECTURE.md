# Polymath Musician Architecture

## Frontend

React/Vite single-page application with route state managed in `src/App.jsx`.

- `pages/StudioPage` flow inside `App.jsx`: grand-piano playback, pedal-aware scheduling, falling notes, and YouTube comparison.
- `pages/GuitarPage.jsx`: guitar tablature/chord playback, fretboard interaction, timeline scrubbing, and ready-to-play imports.
- `pages/EnsemblePage.jsx`: visual teaching, playback, slowing, scrubbing, and ready-to-play lessons for fiddle, banjo, mandolin, dobro, upright bass, ukulele, electric guitar, drum set, and synth keyboard.
- `components/InstrumentTeacherSurface.jsx`: responsive physical-instrument SVG teachers with live press/fret/slide/strike targets.
- `components/MusicUploadPanel.jsx`: plain-language choice between ready-to-play upload and OpenAI PDF translation.
- `components/PdfTranslationPanel.jsx`: allowance/Mcoin choice, progress polling, ETA display, refund/failure messaging, and output download.
- `pages/MarketplacePage.jsx`: format-aware listings, seller fee preview, purchases, and downloads.
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
- Mcoin wallet and seller-withdrawal records;
- marketplace publishing, server-side file validation, exact 10% fee allocation, purchases, and protected downloads;
- YouTube search proxy;
- authenticated OpenAI PDF translation jobs, monthly allowance accounting, Mcoin charging, duplicate-job protection, five-minute ETA extensions, strict structured-output validation, and automatic restoration after failure.
- backend-enforced signup/spending policies, Mcoin vouchers, marketplace coupons, reusable short Friend IDs, Friend ID percentage vouchers, redemption limits, administrator password-reset audits, and hashed 30-day sessions.

## Current persistence

The bundled development backend uses an atomically replaced local JSON database and filesystem uploads. Passwords use salted scrypt hashes and new session records contain SHA-256 token hashes. This keeps setup simple but is not safe for horizontally scaled production traffic.

## Production target

Use a transactional relational database, durable job queue, private object storage, managed secrets, rate limiting, malware scanning, observability, backups, and independently scalable translation workers. Financial actions must execute in database transactions with idempotency keys.
