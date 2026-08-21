# AWS Lightsail deployment

Polymath Musician runs as two Docker Compose services:

- `app`: the built React site and Express API.
- `caddy`: HTTPS termination and reverse proxy.

Persistent user data and Caddy certificates use named Docker volumes.

## Lightsail resource

- Virtual server: Ubuntu 24.04 LTS
- Region: US East
- Plan: Linux with 2 GB RAM and public IPv4
- Networking: attach a static IP; allow ports 80 and 443
- Security: restrict SSH port 22 to the administrator IP when practical

## Deployment

Install Docker Engine and the Docker Compose plugin from Docker's official
Ubuntu repository. Clone the GitHub repository, create the uncommitted
`server/.env`, and start the services:

```bash
git clone https://github.com/ImmanuelKhooKluxKlan/polymath.git
cd polymath
docker compose config
docker compose up -d --build
docker compose ps
```

Keep `PAYPAL_ENV=sandbox` until payment and webhook smoke tests pass. Keep
published MuScriptor weights disabled for commercial traffic unless commercial
permission has been obtained.

## Cloudflare DNS

Create DNS-only A records for `@` and `www`, both targeting the Lightsail
static IPv4. Once Caddy has obtained its HTTPS certificate, Cloudflare proxying
may be enabled with SSL/TLS mode set to **Full (strict)**.

## Updates

```bash
cd ~/polymath
git pull --ff-only
docker compose up -d --build
```

Enable Lightsail automatic snapshots before admitting users. The current data
volume is appropriate for a controlled beta, not a horizontally scaled paid
launch; PostgreSQL and private object storage remain required for that stage.
