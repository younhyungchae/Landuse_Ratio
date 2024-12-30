import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--vision_tower_path', required=True)
args = parser.parse_args()

param_path = os.path.join(args.model_path, 'pytorch_model-00002-of-00002.bin')
param = torch.load(param_path)
vision_tower = torch.load(args.vision_tower_path)

for key, value in vision_tower.items():
    assert key in param.keys()
    param[key] = value

torch.save(param, param_path)