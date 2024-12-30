#!/bin/bash

conda activate geochat
python test.py --model-path ./models/geochat_finetuned_balanced --output_dir ./test.jsonl --image_path ../data/mixed_ground_truth