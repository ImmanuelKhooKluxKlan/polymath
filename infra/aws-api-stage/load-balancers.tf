resource "aws_lb" "ohio" {
  provider           = aws.ohio
  name               = "polymath-api-ohio"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_ohio.id]
  subnets            = aws_subnet.ohio_public[*].id
  idle_timeout       = 60
  tags               = local.tags
}

resource "aws_lb_target_group" "ohio" {
  provider             = aws.ohio
  name                 = "polymath-api-ohio"
  port                 = 3000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.ohio.id
  deregistration_delay = 30

  health_check {
    enabled             = true
    path                = "/api/health"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 20
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  tags = local.tags
}

resource "aws_lb_listener" "ohio" {
  provider          = aws.ohio
  load_balancer_arn = aws_lb.ohio.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ohio.arn
  }
}

resource "aws_lb" "singapore" {
  provider           = aws.singapore
  name               = "polymath-api-singapore"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_singapore.id]
  subnets            = aws_subnet.singapore_public[*].id
  idle_timeout       = 60
  tags               = local.tags
}

resource "aws_lb_target_group" "singapore" {
  provider             = aws.singapore
  name                 = "polymath-api-singapore"
  port                 = 3000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.singapore.id
  deregistration_delay = 30

  health_check {
    enabled             = true
    path                = "/api/health"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 20
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  tags = local.tags
}

resource "aws_lb_listener" "singapore" {
  provider          = aws.singapore
  load_balancer_arn = aws_lb.singapore.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.singapore.arn
  }
}
