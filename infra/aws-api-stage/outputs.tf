output "database_endpoint" {
  value = aws_db_instance.primary.address
}

output "database_secret_arn" {
  value     = aws_db_instance.primary.master_user_secret[0].secret_arn
  sensitive = true
}

output "ohio_load_balancer" {
  value = aws_lb.ohio.dns_name
}

output "singapore_load_balancer" {
  value = aws_lb.singapore.dns_name
}

output "ohio_cluster" {
  value = aws_ecs_cluster.ohio.name
}

output "singapore_cluster" {
  value = aws_ecs_cluster.singapore.name
}

output "initial_worker_counts" {
  value = {
    ohio      = aws_ecs_service.ohio.desired_count
    singapore = aws_ecs_service.singapore.desired_count
  }
}
