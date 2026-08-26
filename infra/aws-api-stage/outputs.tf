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

output "origin_certificate_validation" {
  value = {
    ohio = [for option in aws_acm_certificate.ohio_origin.domain_validation_options : {
      name  = option.resource_record_name
      type  = option.resource_record_type
      value = option.resource_record_value
    }]
    singapore = [for option in aws_acm_certificate.singapore_origin.domain_validation_options : {
      name  = option.resource_record_name
      type  = option.resource_record_type
      value = option.resource_record_value
    }]
  }
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
