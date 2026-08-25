resource "aws_db_subnet_group" "primary" {
  provider   = aws.ohio
  name       = "polymath-database"
  subnet_ids = aws_subnet.ohio_database[*].id
  tags       = merge(local.tags, { Name = "polymath-database" })
}

resource "aws_db_instance" "primary" {
  provider                     = aws.ohio
  identifier                   = "polymath-primary"
  engine                       = "postgres"
  instance_class               = var.database_instance_class
  allocated_storage            = 20
  max_allocated_storage        = 100
  storage_type                 = "gp3"
  storage_encrypted            = true
  db_name                      = "polymath"
  username                     = "polymath_admin"
  manage_master_user_password  = true
  port                         = 5432
  db_subnet_group_name         = aws_db_subnet_group.primary.name
  vpc_security_group_ids       = [aws_security_group.database.id]
  publicly_accessible          = false
  multi_az                     = false
  backup_retention_period      = 7
  backup_window                = "08:00-09:00"
  maintenance_window           = "sun:09:30-sun:10:30"
  auto_minor_version_upgrade   = true
  deletion_protection          = true
  skip_final_snapshot          = false
  final_snapshot_identifier    = "polymath-primary-final"
  copy_tags_to_snapshot        = true
  apply_immediately            = false
  performance_insights_enabled = false
  tags                         = merge(local.tags, { Name = "polymath-primary" })

  lifecycle {
    prevent_destroy = true
  }
}
