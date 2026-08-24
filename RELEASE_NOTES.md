# Polymath Musician 3.0 — Visual Teacher, OpenAI Translation, and AWS-Ready Release

## Added

- Visual teaching studios for five-string fiddle, five-string banjo, mandolin, dobro/resonator guitar, upright double bass, ukulele, electric guitar, drum set, and synth keyboard.
- Separate **Upload Ready-to-Play Sheet** and **Translate to a Ready-to-Play Sheet** experiences.
- PDF validation, authenticated translation jobs, progress stages, 20-minute initial estimate, automatic five-minute estimate extensions, downloadable ready-to-play JSON, duplicate-charge protection, and restoration of payment/allowance on provider failure.
- Monthly translation allowances: 1 for Free and 20 for Pro.
- Choice between allowance usage and a 30-Mcoin translation payment.
- Minimalist Composers catalogue with song, artist, and file-type browsing.
- Permanent verified-buyer reviews, five-star ratings, follows, and public composer profiles.
- Server-side 25% sale fee with 75% composer earnings.
- Percentage-only promotion codes; legacy wallet-credit and fixed-Mcoin codes are retired and cannot be reactivated.
- Responsive phone, tablet, iPad, laptop, and desktop layouts.

## Changed

- Corrected the product name everywhere to **Polymath Musician** while preserving migration from the legacy browser token key.
- Changed the commercial peg to **$1 USD = 10 Mcoins**.
- Changed Pro to **$19.99 USD/month**.
- Replaced direct PDF playback/import with the accountable translation queue.
- Opened cash-out to every account and applied a clearly previewed 25% cash-out fee; requests remain queued for manual review.

## Operational requirement

PDF translation connects directly to the OpenAI Responses API through `OPENAI_API_KEY`. Without the key, the backend rejects the request before deducting Mcoins or allowance.

## Visual Instrument Teacher

- Added physical, illuminated teaching surfaces for fiddle, banjo, mandolin, dobro, upright bass, ukulele, electric guitar, drum set, and synth keyboard.
- Learners can click the actual string/fret/key/drum target, follow live highlights during playback, slow lessons down, scrub the timeline, and upload ready-to-play sheets.
- Renamed the piano tone to **Polymath Musician render** and the piano selector to **Free available to play songs**.
