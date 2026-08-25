terraform {
  required_version = ">= 1.10.0"

  backend "s3" {
    bucket       = "polymath-terraform-state-038223565641"
    key          = "aws-foundation/terraform.tfstate"
    region       = "us-east-2"
    profile      = "polymath-admin"
    encrypt      = true
    use_lockfile = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region  = var.ohio_region
  profile = var.aws_profile

  default_tags {
    tags = local.tags
  }
}

provider "aws" {
  alias   = "singapore"
  region  = var.singapore_region
  profile = var.aws_profile

  default_tags {
    tags = local.tags
  }
}
