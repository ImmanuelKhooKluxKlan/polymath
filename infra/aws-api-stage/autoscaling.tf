resource "aws_appautoscaling_target" "ohio" {
  provider           = aws.ohio
  max_capacity       = 4
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.ohio.name}/${aws_ecs_service.ohio.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ohio_cpu" {
  provider           = aws.ohio
  name               = "polymath-api-us-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ohio.resource_id
  scalable_dimension = aws_appautoscaling_target.ohio.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ohio.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 60
    scale_in_cooldown  = 180
    scale_out_cooldown = 30
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

resource "aws_appautoscaling_policy" "ohio_requests" {
  provider           = aws.ohio
  name               = "polymath-api-us-requests"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ohio.resource_id
  scalable_dimension = aws_appautoscaling_target.ohio.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ohio.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 600
    scale_in_cooldown  = 180
    scale_out_cooldown = 30
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.ohio.arn_suffix}/${aws_lb_target_group.ohio.arn_suffix}"
    }
  }
}

resource "aws_appautoscaling_target" "singapore" {
  provider           = aws.singapore
  max_capacity       = 4
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.singapore.name}/${aws_ecs_service.singapore.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "singapore_cpu" {
  provider           = aws.singapore
  name               = "polymath-api-apac-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.singapore.resource_id
  scalable_dimension = aws_appautoscaling_target.singapore.scalable_dimension
  service_namespace  = aws_appautoscaling_target.singapore.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 60
    scale_in_cooldown  = 180
    scale_out_cooldown = 30
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

resource "aws_appautoscaling_policy" "singapore_requests" {
  provider           = aws.singapore
  name               = "polymath-api-apac-requests"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.singapore.resource_id
  scalable_dimension = aws_appautoscaling_target.singapore.scalable_dimension
  service_namespace  = aws_appautoscaling_target.singapore.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 600
    scale_in_cooldown  = 180
    scale_out_cooldown = 30
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.singapore.arn_suffix}/${aws_lb_target_group.singapore.arn_suffix}"
    }
  }
}
