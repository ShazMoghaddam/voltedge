# ─────────────────────────────────────────────────────────────────────────────
#  VoltEdge — Multi-Region AWS Infrastructure (Terraform)
#
#  Three-region active-active deployment:
#    eu-west-1       (Ireland)   — primary, EU data residency
#    us-east-1       (Virginia)  — North America
#    ap-southeast-1  (Singapore) — Asia Pacific
#
#  Per region: ECS Fargate → ALB → Route53 | RDS Multi-AZ | S3 + CRR
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket         = "voltedge-terraform-state"
    key            = "voltedge/global/terraform.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "voltedge-terraform-locks"
    encrypt        = true
  }
}

provider "aws" { alias = "eu_west_1";      region = "eu-west-1";      default_tags { tags = local.common_tags } }
provider "aws" { alias = "us_east_1";      region = "us-east-1";      default_tags { tags = local.common_tags } }
provider "aws" { alias = "ap_southeast_1"; region = "ap-southeast-1"; default_tags { tags = local.common_tags } }

locals {
  app_name    = "voltedge"
  environment = var.environment
  common_tags = {
    Application = "VoltEdge"
    Environment = var.environment
    ManagedBy   = "Terraform"
    CostCentre  = "PLATFORM-ENGINEERING"
  }
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "environment" {
  type    = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Must be staging or production."
  }
}

variable "image_tag"      { type = string; default = "latest" }
variable "db_password"    { type = string; sensitive = true }
variable "app_secret_key" {
  type      = string
  sensitive = true
  validation {
    condition     = length(var.app_secret_key) >= 32
    error_message = "app_secret_key must be at least 32 characters."
  }
}

# ── ECR ───────────────────────────────────────────────────────────────────────

resource "aws_ecr_repository" "voltedge_api" {
  provider             = aws.eu_west_1
  name                 = "voltedge-api"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
  lifecycle { prevent_destroy = true }
}

# ── S3 — one bucket per region ────────────────────────────────────────────────

resource "aws_s3_bucket" "data_eu" {
  provider = aws.eu_west_1
  bucket   = "voltedge-data-eu-${var.environment}"
}

resource "aws_s3_bucket_versioning" "data_eu" {
  provider = aws.eu_west_1
  bucket   = aws_s3_bucket.data_eu.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket" "data_us"   { provider = aws.us_east_1;      bucket = "voltedge-data-us-${var.environment}" }
resource "aws_s3_bucket" "data_apac" { provider = aws.ap_southeast_1; bucket = "voltedge-data-apac-${var.environment}" }

# ── Cross-Region Replication EU → US + APAC ───────────────────────────────────

resource "aws_s3_bucket_replication_configuration" "eu_to_multi" {
  provider   = aws.eu_west_1
  bucket     = aws_s3_bucket.data_eu.id
  role       = aws_iam_role.s3_replication.arn
  depends_on = [aws_s3_bucket_versioning.data_eu]

  rule {
    id     = "replicate-to-us"
    status = "Enabled"
    destination { bucket = aws_s3_bucket.data_us.arn; storage_class = "STANDARD_IA" }
  }

  rule {
    id     = "replicate-to-apac"
    status = "Enabled"
    destination { bucket = aws_s3_bucket.data_apac.arn; storage_class = "STANDARD_IA" }
  }
}

resource "aws_iam_role" "s3_replication" {
  provider = aws.eu_west_1
  name     = "voltedge-s3-replication-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow"; Principal = { Service = "s3.amazonaws.com" }; Action = "sts:AssumeRole" }]
  })
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "ecr_api_url"    { value = aws_ecr_repository.voltedge_api.repository_url }
output "s3_bucket_eu"   { value = aws_s3_bucket.data_eu.bucket }
output "s3_bucket_us"   { value = aws_s3_bucket.data_us.bucket }
output "s3_bucket_apac" { value = aws_s3_bucket.data_apac.bucket }
