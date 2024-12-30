import os
import json
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from PIL import Image
from tqdm import tqdm
from geochat.conversation import conv_templates, Chat
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import  get_model_name_from_path, process_images, process_images_demo,tokenizer_image_token
from geochat.model.classifier import ChatForClassification

IMAGE_TOKEN_INDEX = -200
CLASSES = [ 'Residential', 
            'Agricultural', 
            'Commercial', 
            'Factory', 
            'Mining', 
            'Power station', 
            'Transportation', 
            'Sports', 
            'Religious', 
            'Water', 
            'Others']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--gpu-id", type=str,default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    return args

cudnn.benchmark = False
cudnn.deterministic = True

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
args = parse_args()

### Load Models ###
model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device, device_map='cuda:0')
model = model.eval()
chat = ChatForClassification(model, image_processor,tokenizer, device='cuda')

def get_all_file_paths(folder_path, recursive=True):
    """
    특정 폴더 내의 모든 파일 경로를 리스트로 반환합니다.

    Parameters:
        folder_path (str): 파일 경로를 검색할 폴더의 경로.
        recursive (bool): True이면 하위 디렉토리까지 포함하여 검색합니다.

    Returns:
        list: 모든 파일 경로를 포함하는 리스트.
    """
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
        if not recursive:
            break
    return file_paths

def foldername_to_ccode(name):
    if name.startswith('PL'):
        return 'PL'
    elif name.startswith('SK'):
        return 'SK'
    elif name.startswith('CZ'):
        return 'CZ'
    elif name.startswith('HU'):
        return 'HU'
    else:
        raise NotImplementedError

paths = get_all_file_paths(args.image_path)

if os.path.exists(args.output_path):
    print("Resume from saved file")
    previous = set([json.loads(line)['path'] for line in open(args.output_path, 'r')])
    to_do = set(paths)
    paths = [path for path in to_do if path not in previous]

for item in tqdm(paths):
    conv = conv_templates['llava_v1'].copy()
    prompt = '<image>\nClassify the given image in one of the following classes. Classes: Residential, Agricultural, Commercial, Factory, Mining, Power station, Sports, Religious, Transportation, Water, Others. \nAnswer in one word or a short phrase.'

    chat.ask(prompt, conv)
    
    img = process_images_demo([Image.open(item).convert('RGB')], image_processor).type(torch.float16).to('cuda')

    generation_kwargs = chat.answer_prepare([conv], [img])
    generation_kwargs['streamer'] = None
    
    image_id = item.split('/')[-1]
    ADM0 = foldername_to_ccode(item.split('/')[-4])
    ADM1 = item.split('/')[-3]
    ADM2 = item.split('/')[-2]

    if args.type == 'gen':
        output = chat.model_generate(kwargs=generation_kwargs)
        with open(args.output_path,'a') as f:
            json.dump({'path':item, 'pred':output}, f)
            f.write('\n')

    elif args.type == 'cls':
        _, probabilities = chat.classify(classes=CLASSES, **generation_kwargs)
        with open(args.output_path,'a') as f:
            json.dump({'path':item, 'image_id':image_id, 'ADM0': ADM0, 'ADM1': ADM1, 'ADM2': ADM2, 'probs':probabilities}, f)
            f.write('\n')

    else:
        raise ValueError

    
