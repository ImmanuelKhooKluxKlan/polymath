resource "aws_security_group" "alb_ohio" {
  provider    = aws.ohio
  name        = "polymath-alb-ohio"
  description = "Public staged API load balancer"
  vpc_id      = aws_vpc.ohio.id

  ingress {
    description = "HTTP staging health and load tests"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "polymath-alb-ohio" })
}

resource "aws_security_group" "ecs_ohio" {
  provider    = aws.ohio
  name        = "polymath-ecs-ohio"
  description = "Only the Ohio load balancer may reach API tasks"
  vpc_id      = aws_vpc.ohio.id

  ingress {
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_ohio.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "polymath-ecs-ohio" })
}

resource "aws_security_group" "alb_singapore" {
  provider    = aws.singapore
  name        = "polymath-alb-singapore"
  description = "Public staged API load balancer"
  vpc_id      = aws_vpc.singapore.id

  ingress {
    description = "HTTP staging health and load tests"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "polymath-alb-singapore" })
}

resource "aws_security_group" "ecs_singapore" {
  provider    = aws.singapore
  name        = "polymath-ecs-singapore"
  description = "Only the Singapore load balancer may reach API tasks"
  vpc_id      = aws_vpc.singapore.id

  ingress {
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_singapore.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "polymath-ecs-singapore" })
}

resource "aws_security_group" "database" {
  provider    = aws.ohio
  name        = "polymath-database"
  description = "PostgreSQL access only from Polymath ECS networks"
  vpc_id      = aws_vpc.ohio.id

  ingress {
    description     = "Ohio API tasks"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_ohio.id]
  }

  ingress {
    description = "Singapore API tasks across regional VPC peering"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.singapore.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "polymath-database" })
}
