#!/bin/bash
source /home/richard8/TweetVerify/bin/activate
python -m cProfile -o output_app.prof src.apps.app

# visualize result
snakeviz output_app.prof