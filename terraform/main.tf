terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_availability_zones" "available" {}

# -----------------------------
# EC2 Security Group
# -----------------------------
resource "aws_security_group" "ec2_sg" {
  name        = "ec2_sg"
  description = "Allow SSH and app inbound traffic"


  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }


  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5001
    to_port     = 5001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# -----------------------------
# RDS Security Group
# -----------------------------
resource "aws_security_group" "rds_sg" {
  name        = "rds_sg"
  description = "Allow PostgreSQL from EC2"

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# -----------------------------
# Get the latest manual snapshot
# -----------------------------
data "aws_db_snapshot" "latest_snapshot" {
  db_instance_identifier = "tweetverify-db" 
  most_recent            = true             
  snapshot_type          = "automated"          
}

# -----------------------------
# RDS PostgreSQL Instance
# -----------------------------
resource "aws_db_instance" "tweetverify_db" {
  identifier              = "tweetverify-db"
  engine                  = "postgres"
  engine_version          = "17.4"
  instance_class          = "db.t3.micro"
  allocated_storage       = 20
  storage_type            = "gp3"
  username                = var.db_username
  password                = var.db_password
  db_name                 = "tweetverify"
  port                    = 5432
  publicly_accessible     = false
  skip_final_snapshot     = false
  vpc_security_group_ids  = [aws_security_group.rds_sg.id]
  multi_az                = false
  backup_retention_period = 1
  deletion_protection     = false

  tags = {
    Name = "tweetverify-db"
  }
}

# -----------------------------
# EC2 Instance
# -----------------------------
resource "aws_instance" "my_ec2" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
    delete_on_termination = true
  }

  user_data = <<-EOF
    #!/bin/bash
    cd /home/ec2-user
    mkdir -p /home/ec2-user/tmp_pip
    export TMPDIR=/home/ec2-user/tmp_pip
    sudo yum update -y
    sudo yum install -y python3 python3-pip git
    sudo yum install -y gcc python3-devel postgresql-devel
    git clone https://github.com/UofT-CSC490-F2025/TweetVerify.git
    cd TweetVerify
    python3 -m venv venv
    source venv/bin/activate
    pip3 install --no-cache-dir -r requirements.txt

    echo "export AWS_ACCESS_KEY_ID=${var.aws_access_key_id}" >> /home/ec2-user/.bashrc
    echo "export AWS_SECRET_ACCESS_KEY=${var.aws_secret_access_key}" >> /home/ec2-user/.bashrc
    echo "export AWS_DEFAULT_REGION=${var.aws_region}" >> /home/ec2-user/.bashrc
    echo "export AWS_ROLE_ARN=${var.aws_role_arn}" >> /home/ec2-user/.bashrc
    echo "export DB_HOST=${aws_db_instance.tweetverify_db.address}" >> /home/ec2-user/.bashrc
    echo "export DB_USER=${var.db_username}" >> /home/ec2-user/.bashrc
    echo "export DB_PASS=${var.db_password}" >> /home/ec2-user/.bashrc
    echo "export DB_NAME=tweetverify" >> /home/ec2-user/.bashrc
    source /home/ec2-user/.bashrc
    source venv/bin/activate
    python3 src/app_wrapper.py
  EOF

  tags = {
    Name = "terraform-ec2"
  }

  depends_on = [aws_db_instance.tweetverify_db]
}
