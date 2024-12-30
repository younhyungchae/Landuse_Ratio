#!/bin/bash

conda activate geochat
python process_sc.py --model-path ./models/geochat_finetuned_balanced --output_dir ./sc_processed.jsonl --image_path ../../geosee/images --batch_size 8