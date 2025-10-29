variable "aws_region" {
  description = "AWS region to deploy resources"
  default     = "us-east-1"
}

variable "ami_id" {
  description = "AMI ID for EC2 instance"
  default     = "ami-0341d95f75f311023"
}

variable "instance_type" {
  description = "EC2 instance type"
  default     = "t3.xlarge"
}

variable "key_name" {
  description = "Name of your existing AWS key pair"
  default     = "abc" 
}

variable "db_username" {
  default = "postgres"
}

variable "db_password" {
  description = "Database password"
  sensitive   = true
}

variable "my_ip_cidr" {
  description = "your ip"
  default     = "0.0.0.0/0"
}

variable "aws_access_key_id" {}
variable "aws_secret_access_key" {
  sensitive   = true
}
variable "aws_role_arn" {}