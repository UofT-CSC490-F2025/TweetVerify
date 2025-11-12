#!/bin/bash
source /home/richard8/TweetVerify/bin/activate
python -m cProfile -o output_train.prof train.py

# visualize result
snakeviz output_train.prof