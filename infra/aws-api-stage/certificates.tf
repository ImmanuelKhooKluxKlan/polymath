resource "aws_acm_certificate" "ohio_origin" {
  provider          = aws.ohio
  domain_name       = "api-us-origin.polymathmusician67.com"
  validation_method = "DNS"
  tags              = local.tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_acm_certificate" "singapore_origin" {
  provider          = aws.singapore
  domain_name       = "api-apac-origin.polymathmusician67.com"
  validation_method = "DNS"
  tags              = local.tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_acm_certificate_validation" "ohio_origin" {
  provider                = aws.ohio
  certificate_arn         = aws_acm_certificate.ohio_origin.arn
  validation_record_fqdns = [for option in aws_acm_certificate.ohio_origin.domain_validation_options : option.resource_record_name]
}

resource "aws_acm_certificate_validation" "singapore_origin" {
  provider                = aws.singapore
  certificate_arn         = aws_acm_certificate.singapore_origin.arn
  validation_record_fqdns = [for option in aws_acm_certificate.singapore_origin.domain_validation_options : option.resource_record_name]
}
