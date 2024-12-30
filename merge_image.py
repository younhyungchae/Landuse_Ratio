import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

data = [json.loads(line) for line in open('./subregions.jsonl')]

ccodes = ['PL','HU','SK','CZ']

targets = {'PL':'Miasto Kraków',
           'HU':'Hajdú-Bihar',
           'SK':'Košický kraj',
           'CZ':'Jihomoravský kraj'}

for ccode in ccodes:
    if ccode in ['HU', 'PL']:
        continue
    target_data = [item for item in data if item['path'].split('/')[3].startswith(ccode)]
    coordinates = [item['path'].split('/')[-1].replace('.png','').split('_') for item in target_data]

    height_min = min([int(item[0]) for item in coordinates])
    height_max = max([int(item[0]) for item in coordinates])
    width_min = min([int(item[1]) for item in coordinates])
    width_max = max([int(item[1]) for item in coordinates])

    image_path = f'../data/subregions_images/{ccode}_{targets[ccode]}'
    images = os.listdir(image_path)

    new_image = Image.new(size=(int(256*(width_max-width_min+1)), int(256*(height_max-height_min+1))), mode='RGB')

    for item in tqdm(target_data):
        coordinate = item['path'].split('/')[-1].replace('.png','').split('_')
        x = int(coordinate[0])-height_min
        y = int(coordinate[1])-width_min
        Image.Image.paste(new_image, Image.open(item['path']), (int(256*y), int(256*x)))

    new_image.save(f'../plots/{ccode}_full.png')