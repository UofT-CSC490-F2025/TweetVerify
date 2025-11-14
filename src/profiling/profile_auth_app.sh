#!/bin/bash
source /home/richard8/TweetVerify/bin/activate
python -m cProfile -o output_auth_app.prof src.apps.auth_app

# visualize result
snakeviz output_auth_app.prof