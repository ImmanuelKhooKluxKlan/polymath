data "aws_iam_policy_document" "ecs_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  provider           = aws.ohio
  name               = "polymath-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  provider   = aws.ohio
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  provider           = aws.ohio
  name               = "polymath-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "ecs_task" {
  statement {
    sid    = "ReadRuntimeAndDatabaseSecrets"
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
    ]
    resources = [
      data.aws_secretsmanager_secret.runtime.arn,
      aws_db_instance.primary.master_user_secret[0].secret_arn,
    ]
  }

  statement {
    sid    = "UseJobQueue"
    effect = "Allow"
    actions = [
      "sqs:DeleteMessage",
      "sqs:ReceiveMessage",
      "sqs:SendMessage",
    ]
    resources = [data.terraform_remote_state.foundation.outputs.job_queue_arn]
  }

  statement {
    sid    = "SendRegistrationVerification"
    effect = "Allow"
    actions = [
      "ses:SendEmail",
      "ses:SendRawEmail",
      "sns:Publish",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "SynthesizeNaturalTeacherSpeech"
    effect    = "Allow"
    actions   = ["polly:SynthesizeSpeech"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "ecs_task" {
  provider = aws.ohio
  name     = "polymath-ecs-runtime"
  role     = aws_iam_role.ecs_task.id
  policy   = data.aws_iam_policy_document.ecs_task.json
}
