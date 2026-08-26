# Polymath staged AWS API

This stack creates the next backend without moving production traffic:

- ECS Fargate API services in Ohio and Singapore, autoscaling from one to four
  workers per region by CPU and load-balancer request count.
- One private, encrypted, Multi-AZ PostgreSQL database in Ohio.
- Cross-region private networking from Singapore to the database.
- One HTTPS load balancer per region, with TLS 1.2/1.3 certificates and HTTP to
  HTTPS redirects.
- Encrypted logs, IAM roles, SQS access, and Secrets Manager startup.

The regional staging origins are:

- `api-us-origin.polymathmusician67.com`
- `api-apac-origin.polymathmusician67.com`

GitHub Actions owns application releases and ECS task revisions. Terraform owns
the service infrastructure and intentionally ignores `task_definition` and
autoscaler-managed `desired_count` changes so an infrastructure apply cannot
roll back a deployed release or fight the autoscaler.

The existing Lightsail API remains production until Cloudflare Load Balancing
has healthy HTTPS monitors, a pool for each origin, automatic failover, and the
required token permissions. Do not manually repoint the production `api` DNS
record before those controls are active.

Never put passwords in Terraform variables. Runtime values belong in the
polymath/api-runtime AWS Secrets Manager secret; RDS manages its own password.
