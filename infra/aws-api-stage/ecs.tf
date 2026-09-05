locals {
  common_environment = [
    { name = "NODE_ENV", value = "production" },
    { name = "PORT", value = "3000" },
    { name = "CLIENT_ORIGIN", value = var.client_origin },
    { name = "CLIENT_ORIGINS", value = var.client_origins },
    { name = "POLYMATH_DATA_DIR", value = "/tmp/polymath-data" },
    { name = "PGHOST", value = aws_db_instance.primary.address },
    { name = "PGPORT", value = tostring(aws_db_instance.primary.port) },
    { name = "PGDATABASE", value = aws_db_instance.primary.db_name },
    { name = "DATABASE_SSL", value = "true" },
    { name = "DATABASE_SSL_REJECT_UNAUTHORIZED", value = "true" },
    { name = "DATABASE_SSL_CA_PATH", value = "/app/aws-rds-global-bundle.pem" },
    { name = "DATABASE_POOL_MAX", value = "10" },
    { name = "JOB_QUEUE_URL", value = data.terraform_remote_state.foundation.outputs.job_queue_url },
    { name = "JOB_QUEUE_REGION", value = "us-east-2" },
    { name = "ARTIFACT_S3_BUCKET", value = var.artifact_bucket },
    { name = "ARTIFACT_S3_ENDPOINT", value = var.artifact_endpoint },
    { name = "ARTIFACT_S3_REGION", value = "auto" },
    { name = "ARTIFACT_S3_FORCE_PATH_STYLE", value = "false" },
    { name = "AWS_SECRET_REGION", value = "us-east-2" },
    { name = "TEACHER_TTS_ENABLED", value = "true" },
    { name = "AWS_RUNTIME_SECRET_ARN", value = data.aws_secretsmanager_secret.runtime.arn },
    { name = "AWS_RDS_SECRET_ARN", value = aws_db_instance.primary.master_user_secret[0].secret_arn },
  ]
}

resource "aws_cloudwatch_log_group" "ohio" {
  provider          = aws.ohio
  name              = "/ecs/polymath-api-us"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "singapore" {
  provider          = aws.singapore
  name              = "/ecs/polymath-api-apac"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_ecs_cluster" "ohio" {
  provider = aws.ohio
  name     = "polymath-us"
  tags     = local.tags
}

resource "aws_ecs_cluster" "singapore" {
  provider = aws.singapore
  name     = "polymath-apac"
  tags     = local.tags
}

resource "aws_ecs_task_definition" "ohio" {
  provider                 = aws.ohio
  family                   = "polymath-api-us"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "scratch"
  }

  container_definitions = jsonencode([{
    name      = "api"
    image     = "${data.terraform_remote_state.foundation.outputs.ohio_ecr_repository_url}:${var.image_tag}"
    essential = true
    portMappings = [{
      containerPort = 3000
      hostPort      = 3000
      protocol      = "tcp"
      appProtocol   = "http"
    }]
    environment = concat(local.common_environment, [
      { name = "APP_REGION", value = "us-east-2" },
    ])
    readonlyRootFilesystem = true
    mountPoints = [{
      sourceVolume  = "scratch"
      containerPath = "/tmp"
      readOnly      = false
    }]
    linuxParameters = {
      initProcessEnabled = true
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ohio.name
        "awslogs-region"        = "us-east-2"
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
  tags = local.tags
}

resource "aws_ecs_task_definition" "singapore" {
  provider                 = aws.singapore
  family                   = "polymath-api-apac"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "scratch"
  }

  container_definitions = jsonencode([{
    name      = "api"
    image     = "${data.terraform_remote_state.foundation.outputs.singapore_ecr_repository_url}:${var.image_tag}"
    essential = true
    portMappings = [{
      containerPort = 3000
      hostPort      = 3000
      protocol      = "tcp"
      appProtocol   = "http"
    }]
    environment = concat(local.common_environment, [
      { name = "APP_REGION", value = "ap-southeast-1" },
    ])
    readonlyRootFilesystem = true
    mountPoints = [{
      sourceVolume  = "scratch"
      containerPath = "/tmp"
      readOnly      = false
    }]
    linuxParameters = {
      initProcessEnabled = true
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.singapore.name
        "awslogs-region"        = "ap-southeast-1"
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
  tags = local.tags
}

resource "aws_ecs_service" "ohio" {
  provider                           = aws.ohio
  name                               = "polymath-api-us"
  cluster                            = aws_ecs_cluster.ohio.id
  task_definition                    = aws_ecs_task_definition.ohio.arn
  desired_count                      = var.ohio_desired_count
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  health_check_grace_period_seconds  = 60
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  enable_execute_command             = false

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = aws_subnet.ohio_public[*].id
    security_groups  = [aws_security_group.ecs_ohio.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ohio.arn
    container_name   = "api"
    container_port   = 3000
  }

  lifecycle {
    # Application releases are promoted by GitHub Actions. Terraform owns the
    # service infrastructure, but must not roll a deployed task revision back.
    ignore_changes = [desired_count, task_definition]
  }
  depends_on = [aws_lb_listener.ohio, aws_iam_role_policy_attachment.ecs_execution]
  tags       = local.tags
}

resource "aws_ecs_service" "singapore" {
  provider                           = aws.singapore
  name                               = "polymath-api-apac"
  cluster                            = aws_ecs_cluster.singapore.id
  task_definition                    = aws_ecs_task_definition.singapore.arn
  desired_count                      = var.singapore_desired_count
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  health_check_grace_period_seconds  = 90
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  enable_execute_command             = false

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = aws_subnet.singapore_public[*].id
    security_groups  = [aws_security_group.ecs_singapore.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.singapore.arn
    container_name   = "api"
    container_port   = 3000
  }

  lifecycle {
    # Application releases are promoted by GitHub Actions. Terraform owns the
    # service infrastructure, but must not roll a deployed task revision back.
    ignore_changes = [desired_count, task_definition]
  }
  depends_on = [aws_lb_listener.singapore, aws_iam_role_policy_attachment.ecs_execution]
  tags       = local.tags
}
