variable "aws_profile" {
  description = "Temporary/local AWS CLI profile. Set to null in CI."
  type        = string
  default     = "polymath-admin"
  nullable    = true
}

variable "ohio_region" {
  type    = string
  default = "us-east-2"
}

variable "singapore_region" {
  type    = string
  default = "ap-southeast-1"
}

variable "github_repository" {
  description = "Immutable GitHub OIDC owner/repository subject, including numeric IDs."
  type    = string
  default = "ImmanuelKhooKluxKlan@293514804/polymath@1317923756"
}

variable "budget_email" {
  description = "Address receiving AWS actual and forecast budget alerts."
  type        = string
  default     = "immanuelshippingexpress@gmail.com"
}

variable "monthly_budget_usd" {
  type    = number
  default = 650
}

locals {
  name = "polymath"
  tags = {
    Application = "Polymath Musician"
    ManagedBy   = "Terraform"
    Environment = "production"
  }
}
