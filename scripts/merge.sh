#!/bin/bash

conda activate gis

python scripts/merge_lora_weights.py    --model-path ./models/geochat_ratio_label_lora/ \
                                        --model-base ./models/geochat_base \
                                        --save-model-path ./models/geochat_ratio_label

python geochat/replace_vision_tower.py  --model_path ./models/geochat_ratio_label \
                                        --vision_tower_path ./models/vision_tower.bin
