variable "aws_profile" {
  description = "Temporary/local AWS CLI profile. Set to null in CI."
  type        = string
  default     = "polymath-admin"
  nullable    = true
}

variable "project_name" {
  type    = string
  default = "polymath"
}

variable "image_tag" {
  description = "Initial image tag. Services remain at zero until an immutable GitHub SHA is deployed."
  type        = string
  default     = "bootstrap"
}

variable "ohio_desired_count" {
  type    = number
  default = 0
  validation {
    condition     = var.ohio_desired_count >= 0 && var.ohio_desired_count <= 10
    error_message = "Ohio desired count must stay between 0 and 10 during the staged rollout."
  }
}

variable "singapore_desired_count" {
  type    = number
  default = 0
  validation {
    condition     = var.singapore_desired_count >= 0 && var.singapore_desired_count <= 10
    error_message = "Singapore desired count must stay between 0 and 10 during the staged rollout."
  }
}

variable "task_cpu" {
  type    = number
  default = 512
}

variable "task_memory" {
  type    = number
  default = 1024
}

variable "database_instance_class" {
  type    = string
  default = "db.t4g.small"
}

variable "runtime_secret_name" {
  type    = string
  default = "polymath/api-runtime"
}

variable "artifact_bucket" {
  type    = string
  default = "polymath-artifacts"
}

variable "artifact_endpoint" {
  type    = string
  default = "https://9c8b0c2fbbe89d2705bd0a30af9c3e32.r2.cloudflarestorage.com"
}

variable "client_origin" {
  type    = string
  default = "https://polymathmusician67.com"
}

variable "client_origins" {
  type    = string
  default = "https://polymath-musician.pages.dev,https://scaling-preview.polymath-musician.pages.dev"
}

locals {
  tags = {
    Application = "Polymath Musician"
    ManagedBy   = "Terraform"
    Environment = "staging"
  }
}
