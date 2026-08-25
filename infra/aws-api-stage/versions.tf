terraform {
  required_version = ">= 1.10.0"

  backend "s3" {
    bucket       = "polymath-terraform-state-038223565641"
    key          = "aws-api-stage/terraform.tfstate"
    region       = "us-east-2"
    profile      = "polymath-admin"
    encrypt      = true
    use_lockfile = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.61"
    }
  }
}

provider "aws" {
  region  = "us-east-2"
  profile = var.aws_profile
}

provider "aws" {
  alias   = "ohio"
  region  = "us-east-2"
  profile = var.aws_profile
}

provider "aws" {
  alias   = "singapore"
  region  = "ap-southeast-1"
  profile = var.aws_profile
}
