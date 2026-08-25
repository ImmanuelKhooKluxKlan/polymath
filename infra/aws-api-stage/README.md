# Polymath staged AWS API

This stack creates the next backend without moving production traffic:

- ECS Fargate API services in Ohio and Singapore.
- One private encrypted PostgreSQL database in Ohio.
- Cross-region private networking from Singapore to the database.
- One public load balancer per region for health/load testing.
- Encrypted logs, IAM roles, SQS access, and Secrets Manager startup.

Both ECS services start with desired count zero. This is intentional: the
infrastructure can be inspected before an image is deployed or workers begin
billing. The existing Lightsail API remains production until migration and
failover tests pass.

Never put passwords in Terraform variables. Runtime values belong in the
polymath/api-runtime AWS Secrets Manager secret; RDS manages its own password.
