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


resource "aws_security_group" "ec2_sg" {
  name        = "ec2_sg"
  description = "Allow SSH and HTTP inbound traffic"


  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  #  HTTP (80)
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 5000
    to_port     = 5000
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

# create ec2
resource "aws_instance" "my_ec2" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]

  root_block_device {
    volume_size = 50   a
    volume_type = "gp3" 
    delete_on_termination = true
  }

  user_data = <<-EOF
    #!/bin/bash
    cd /home/ec2-user
    sudo yum update -y
    sudo yum install -y python3 python3-pip git
    git clone https://github.com/UofT-CSC490-F2025/TweetVerify.git
    cd TweetVerify
    mkdir -p /home/ec2-user/tmp_pip
    export TMPDIR=/home/ec2-user/tmp_pip
    pip3 install --no-cache-dir -r requirements.txt
    nohup sudo python3 -m src.app > app.log 2>&1 &
  EOF

  tags = {
    Name = "terraform-ec2"
  }
}
