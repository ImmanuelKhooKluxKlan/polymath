data "terraform_remote_state" "foundation" {
  backend = "s3"
  config = {
    bucket       = "polymath-terraform-state-038223565641"
    key          = "aws-foundation/terraform.tfstate"
    region       = "us-east-2"
    profile      = var.aws_profile
    use_lockfile = true
  }
}

data "aws_caller_identity" "current" {
  provider = aws.ohio
}

data "aws_availability_zones" "ohio" {
  provider = aws.ohio
  state    = "available"
}

data "aws_availability_zones" "singapore" {
  provider = aws.singapore
  state    = "available"
}

data "aws_secretsmanager_secret" "runtime" {
  provider = aws.ohio
  name     = var.runtime_secret_name
}
