# Polymath Artist Campaign Launch System

Status: built for local verification; do not publish a campaign until the launch checklist passes.

## 1. The salesperson's brief to engineering

The fastest credible growth loop is not “show every Polymath feature.” It is:

1. A fan opens one artist-owned link.
2. The fan recognizes a song they already care about.
3. One tap prepares the piano.
4. The fan hears or plays a 10–45 second, human-verified challenge.
5. Polymath gives a score the fan can share.
6. The shared link opens the same challenge for a friend.
7. Only after the first win does Polymath invite the fan to Musician.

Sales therefore asked engineering for:

- a direct campaign link with social-preview metadata;
- no sign-in and no payment card before the first attempt;
- an artist name, song name, cover, link, and creator referral code;
- exact approved piano notes rather than public raw transcription output;
- a human-quality badge backed by a real QA record;
- a share card that carries the campaign and score but never the recording;
- attribution through registration and checkout;
- a per-campaign visitor → loaded → attempted → completed → shared → checkout → activated funnel;
- a hard rights and quality gate that sales cannot accidentally bypass;
- pause, archive, edit, and delete controls in the protected admin console.

The business goal is to buy as little attention as possible. The artist supplies trusted distribution; Polymath supplies an interactive experience worth sharing.

## 2. Customer journey

```text
Artist post / QR / message
          │
          ▼
  /c/artist-song
  social title + cover
          │
          ▼
Human-verified challenge
  no account, no card
          │
     ┌────┴────┐
     ▼         ▼
 Hear it     Play it
     └────┬────┘
          ▼
     Score / proof
       │       │
       ▼       ▼
    Share    Musician
       │       │
       ▼       ▼
 New challenger  PayPal checkout
                       │
                       ▼
               Attributed activation
```

The server generates a separate, clipped JSON copy of the approved JSON or MIDI asset. It shifts the selected preview to time zero while preserving pitch, both hands, velocity, note holds, and pedal transitions. The original upload remains private; notes and extra fields outside the approved time window never enter the public response or browser.

## 3. Publication gate

A campaign can be a private draft with missing information. The server refuses `published` status until all of these are true:

- artist, song title, and unique URL slug exist;
- a valid ready-to-play JSON or standard MIDI file exists;
- the public excerpt is between 10 and 45 seconds;
- music rights are confirmed;
- the rights holder or approving artist is recorded;
- the permission or licence basis is recorded;
- artist approval is confirmed;
- a human pianist has verified the result;
- the verification notes say what was checked;
- QA is at least 80/100;
- an optional end time is later than the launch time.

“Human verified” must mean someone listened and checked pitches, timing, holds, dynamics, pedal, playability, and whether the selected excerpt begins and ends cleanly. It is not a marketing label that may be added to unreviewed AI output.

The server invalidates stale approvals. Replacing the song or changing the excerpt window clears pianist QA, verification notes, and artist approval. Replacing the song or changing the artist/song identity also clears the recorded rights confirmation. Changing public creative copy, the cover, target score, or URL returns the campaign to draft and requires artist approval again. Sending `status=published` in the same API request cannot bypass these resets.

## 4. Admin workflow

Open **Admin console → Growth → Playable campaigns**.

1. Select **New campaign**.
2. Add the artist, song, headline, URL slug, referral code, and optional artist website.
3. Upload the rights-cleared ready-to-play JSON/MIDI and optional cover.
4. Select the excerpt start and 10–45 second length.
5. Save a private draft.
6. Select **Preview**. The administrator-only player opens the exact generated excerpt without publishing it, recording campaign analytics, or enabling sharing.
7. Complete human verification, QA, permission, and artist-approval evidence.
8. Save again and confirm the card says **Launch-ready**.
9. Publish, then copy the public campaign link.
10. Pause immediately if the artist withdraws permission or quality is questioned.

Editing a published campaign saves it back as a private draft. That forces changed content through the gate again.

## 5. Technical map

### State

Campaign metadata is part of the shared Polymath state document in `artistCampaigns`. Original song uploads, generated public excerpts, and cover bytes live in the configured artifact store (local disk for development, S3-compatible object storage in production). The original JSON/MIDI key is admin-only. Public links can retrieve only the generated, bounded JSON excerpt.

No campaign is seeded or automatically published. Existing copyrighted training material is not turned into a campaign.

### Public API

- `GET /api/artist-campaigns` — currently live campaigns only.
- `GET /api/artist-campaigns/:slug` — safe public metadata for one live campaign.
- `GET /api/artist-campaigns/:slug/song` — generated excerpt JSON, only while the campaign is live; it never returns the original upload.
- `GET /api/artist-campaigns/:slug/cover` — cover image, only while live.
- `GET /c/:slug` — production HTML with Open Graph/Twitter metadata, then the React challenge route.

Internal asset keys, permission notes, user data, and admin fields never appear in the public campaign response.

### Admin API

- `GET /api/admin/artist-campaigns`
- `GET /api/admin/artist-campaigns/preview/:slug`
- `GET /api/admin/artist-campaigns/:id/preview-song`
- `POST /api/admin/artist-campaigns`
- `PATCH /api/admin/artist-campaigns/:id`
- `DELETE /api/admin/artist-campaigns/:id`

All admin campaign routes require a valid signed-in administrator session. Uploads accept one JSON/MIDI song and one PNG/JPEG/WebP cover, each no larger than 8 MB. Private preview attempts are deliberately excluded from campaign attribution and funnel evidence.

### Attribution

The browser remembers the first valid campaign touch for 30 days. Checkout sends only campaign ID, slug, and referral code. The server accepts attribution only when the campaign is still live and replaces browser claims with canonical server data. Subscription activation remains a trusted server event; a browser cannot manufacture attributed activation value. This value is the selected plan price at activation, not proof that cash has settled.

The dashboard's creator commission is an estimate:

```text
estimated commission = attributed subscription activation value × campaign percentage
```

It is not an automatic payout, accounting record, lifetime-value estimate, or guarantee.

### Privacy

Product events do not store recordings, filenames, song titles, messages, IP addresses, emails, or phone numbers. Shared score cards contain a score and public campaign link, not the player's recording.

## 6. Funnel definitions

| Stage | Event | Meaning |
|---|---|---|
| Viewed | `campaign_viewed` | Public metadata rendered |
| Loaded | `campaign_song_loaded` | Approved playable data parsed |
| Attempted | `campaign_attempt_started` | User started a measured attempt |
| Completed | `campaign_attempt_completed` | A score was produced |
| Shared | `campaign_shared` | Share sheet opened or link copied |
| Checkout | `checkout_started` | Authenticated PayPal checkout began with valid attribution |
| Activated | `subscription_activated` | Server verified subscription activation; this is not a cash-settlement event |

Use unique people, not raw clicks, when comparing stages. A high view count with a low load rate suggests performance or file problems. A good load rate with a low attempt rate suggests unclear preparation or weak song recognition. A good completion rate with weak sharing suggests the result is not emotionally valuable or the card is poor. A good checkout rate with weak payment suggests pricing, trust, or PayPal friction.

## 7. Launch decision thresholds

These are initial operating hypotheses, not guaranteed universal benchmarks:

- zero unresolved rights or artist-approval questions;
- 100% successful song loads across the agreed test devices;
- no severe timing, stuck-note, missing-note, pedal, or mobile-audio defects;
- QA at least 80/100; aim for 90+ before paid promotion;
- at least 20 external test users before advertising;
- manually inspect every shared link on iPhone, Android, tablet, and desktop;
- do not increase promotion when failures or complaints rise.

The first campaign is an experiment. Change one major variable at a time (artist, excerpt, headline, or offer) so the evidence remains understandable.

## 8. Proposed first $500 campaign budget

Do not spend this until the rights-cleared challenge passes QA and 20 external tests.

| Use | Maximum | Reason |
|---|---:|---|
| Artist/creator launch assets | $100 | One clear vertical demo, link visual, and QR asset |
| Small creator tests | $200 | Several small tests are safer than one unproven large placement |
| Retargeting/paid test | $150 | Only after organic visitors load, attempt, and share |
| Contingency | $50 | Fix creative or placement problems without expanding budget |

Stop spending if song-load reliability is below 99%, verified complaints appear, or the campaign has no meaningful attempt/share signal. Scale only the exact source and creative that produces measured completed attempts—not impressions alone.

## 9. Pre-market checklist

- [ ] Written rights/permission evidence is stored outside the public UI.
- [ ] Artist approved the cover, copy, excerpt, and destination link.
- [ ] A pianist verified pitch, rhythm, holds, velocity, pedal, and musical feel.
- [ ] QA is 80+ (preferably 90+).
- [ ] The direct link has the correct title, description, image, and artist.
- [ ] Challenge loads from a fresh browser with no account.
- [ ] Screen keys, computer keys, and MIDI input were tested where supported.
- [ ] iPhone and Android audio unlock works after one explicit tap.
- [ ] Falling notes touch the keys at the intended strike time.
- [ ] Share link opens the same campaign and carries the score.
- [ ] Sign-up and checkout preserve canonical campaign attribution.
- [ ] Admin funnel receives test events but no personal or song data.
- [ ] Pause and delete behavior were tested.
- [ ] PayPal is tested in sandbox before any live charge.
- [ ] Support contact and refund terms are visible and current.

### Repeatable engineering checks

Run these from the repository root before asking external testers to open a campaign:

```powershell
npm run lint
npm run test:performance
npm --prefix server test
npm run qa:artist-campaign
```

The visual QA command creates an isolated temporary database, publishes only a local fixture, and captures the public challenge at exact 390 × 844 and 1440 × 1000 viewports. It also opens the protected Growth dashboard and its campaign editor. The output reports the screenshot paths, the real browser viewport width, document scroll width, and any unintended horizontal overflow. It does not alter the normal local database or publish a real campaign.

## 10. Rollback

If anything is wrong, select **Pause**. New origin requests for public metadata and both campaign assets then return 404 while the draft data remains available to the administrator. Song and cover responses are marked `private, no-store`; campaign metadata has a 15-second revalidation window. A social network may independently retain a preview it already scraped, and data already delivered to a visitor cannot be recalled. Fix the campaign, repeat human verification, save as a draft, and publish again only after every blocker is gone.
