# TweetVerify
AI Detection Tool

git clone https://github.com/UofT-CSC490-F2025/TweetVerify.git
cd TweetVerify/terraform
terraform apply \
  -var="db_password=YOURPASSWORD" \
  -var="aws_access_key_id=YOUR AWSAccessKeyID" \
  -var="aws_secret_access_key=YOUR AWSSecretAccessKey" 
terraform init
terraform apply
