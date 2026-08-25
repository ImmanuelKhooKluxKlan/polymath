# AWS foundation

This Terraform stack creates only the low/near-zero idle-cost foundation:

- a USD 650 monthly budget with USD 200, 400, 650 actual alerts and a USD 650 forecast alert;
- encrypted Ohio SQS job and dead-letter queues;
- immutable ECR repositories in Ohio and Singapore;
- a GitHub OIDC deployment role restricted to this repository's `main` branch.

It deliberately does **not** create RDS, ECS tasks, load balancers, NAT gateways,
or other continuously billed compute. Those are activated only after Cloudflare
shared storage and migration smoke tests are ready.

```powershell
$tf = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Hashicorp.Terraform_Microsoft.Winget.Source_8wekyb3d8bbwe\terraform.exe"
& $tf init
& $tf fmt -check
& $tf validate
& $tf plan -out foundation.tfplan
& $tf apply foundation.tfplan
```

Terraform state is stored in the private, encrypted and versioned
`polymath-terraform-state-038223565641` S3 bucket. Native S3 locking prevents
two applies from changing the foundation simultaneously.
