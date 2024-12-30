#!/bin/bash

conda activate geochat
python evaluate.py --model-path ../geochat_ratio_label --image-path ../../geosee/images/PL_16_9549 --output-path ../pl.jsonl --type cls
