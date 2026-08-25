output "github_deploy_role_arn" {
  value = aws_iam_role.github_deploy.arn
}

output "job_queue_url" {
  value = aws_sqs_queue.jobs.url
}

output "job_queue_arn" {
  value = aws_sqs_queue.jobs.arn
}

output "dead_letter_queue_url" {
  value = aws_sqs_queue.jobs_dead_letter.url
}

output "ohio_ecr_repository_url" {
  value = aws_ecr_repository.api_ohio.repository_url
}

output "singapore_ecr_repository_url" {
  value = aws_ecr_repository.api_singapore.repository_url
}

output "monthly_budget_usd" {
  value = var.monthly_budget_usd
}
