# Migration Notes

## Branding

The canonical name is now **Polymath Musician**. The frontend automatically reads the former `polymath_muscian_auth_token` browser key once and migrates it to `polymath_musician_auth_token`, so existing signed-in users are not needlessly logged out.

## Currency

All newly exposed products use USD:

- 50 Mcoins = $5
- 100 Mcoins = $10
- 300 Mcoins = $30
- Pro = $19.99/month
- PDF translation = 30 Mcoins ($3 equivalent)

Existing Mcoin balances remain unchanged because Mcoins are stored as units, not as a fiat amount. Old PayPal order records should be retained for audit purposes, but new purchases must use the new product IDs and USD catalogue.

## Marketplace fee

New purchases allocate 10% of the Mcoin price to Polymath Musician and 90% to the seller. Historical purchase records in the bundled local development database receive derived fee fields for display consistency; production migration should preserve original accounting evidence and use a reviewed ledger migration.

## PDF conversion

The former direct `/api/score-import` route is retired. Clients must use the authenticated `/api/score-translations` flow. Configure `OPENAI_API_KEY` and the selected `OPENAI_MODEL` in the backend environment before enabling the feature publicly. The legacy converter URL variables are no longer used.
