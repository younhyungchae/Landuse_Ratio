&#8251; This code is based on [GeoChat](https://github.com/mbzuai-oryx/GeoChat.git) repository. Refer that repository for more detailed informations.
# Landuse Ratio Estimation
This project estimates the landuse ratio of the single satellite image. This project is based on GeoChat. The model is trained to directly estimate ratio of landuse in images. The process is modeled as classification. The classification probabilities directly maps to ratio estimation.

The main difference in code implementation is [geochat/model/classifier.py](https://github.com/younhyungchae/Landuse_Ratio/blob/a30faa2690b3d306bff5aa00368ce8faf9397045/geochat/model/classifier.py) and [geochat/train/train_classifier.py](https://github.com/younhyungchae/Landuse_Ratio/blob/a30faa2690b3d306bff5aa00368ce8faf9397045/geochat/train/train_classifier.py).

|Classes|
|-----------|
|Residential|
|Agricultural|
|Commercial|
|Factory|
|Mining|
|Power station|
|Transportation|
|Sports|
|Religious|
|Water|
|Others|

The model is trained with [SeasoNet](https://zenodo.org/records/5850307) dataset.

Base classes in SeasoNet dataset is mapped as following. Classes not included below are considered as 'Others'.
```
class_mapping = {'Residential':['Continuous urban fabric', 'Discontinuous urban fabric'],
                    'Agricultural':['Permanently irrigated land','Rice fields','Vineyards','Fruit trees and berry plantations','Olive groves','Complex cultivation patterns','Annual crops associated with permanent crops','Land principally occupied by agriculture, with significant areas of natural vegetation','Non-irrigated arable land'],                
                    'Commercial': ['Industrial or commercial units'],
                    'Factory':[],
                    'Mining':['Mineral extraction sites'],
                    'Power station':[],
                    'Transportation':['Port areas','Airports','Road and rail networks and associated land'],
                    'Sports':['Sport and leisure facilities'],
                    'Religious':[],
                    'Water':['Sea and ocean','Water courses','Water bodies','Coastal lagoons']}
```

## Pretrained Model
Pretrained Model: [landuse_ratio_classification_geochat](https://huggingface.co/YounhyungChae/landuse_ratio_classification_geochat)

## Dataset
|key|description|
|----------|-----------------------|
|id| image id, not necessary|
|image| image path, required|
|converations| prompt and label, required|
|label| dictionary in the format of  {class: ratio_label}, required|

### Dataset Example
```
[{"id": "33UUQ_20180703T101029_48_789966_13_118461", "image": "33UUQ_20180703T101029_48_789966_13_118461.jpg", "conversations": [{"from": "human", "value": "<image>\nClassify the given image in one of the following classes. Classes: Residential, Agricultural, Commercial, Factory, Mining, Power station, Sports, Religious, Transportation, Water, Others. \nAnswer in one word or a short phrase.", "label": {"Residential": 0.07680555555555556, "Agricultural": 0.14381944444444444, "Commercial": 0, "Factory": 0, "Mining": 0, "Power station": 0, "Transportation": 0, "Sports": 0, "Religious": 0, "Water": 0, "Others": 0.26979166666666665}}]}]
```

## Training
1. Download [Geochat Base Model](https://huggingface.co/MBZUAI/geochat-7B)
  ```
git lfs install
git clone https://huggingface.co/MBZUAI/geochat-7B
  ```
2. Remove Vision Tower from Geochat.
   
  &#8251; You must remove vision tower parameter as well as mapping dictionary in *pytorch_model.bin.index.json*
  
3. Train with DeepSpeed ZeRO-3: finetune_lora.sh: `bash scripts/finetune_lora.sh`

4. Merge LoRA adapter with base model: `bash scripts/merge.sh`
   
## Inference
Following command finds all images in image path and append the estimation result to output path.

Output path should be jsonline file.

`python evaluate.py --model-path [MODEL_PATH] --image-path [IMAGE_PATH] --output-path [OUTPUT_PATH] --type cls`

The code will automatically detects images not processed. (Does not perform redundant estimation.)

## Visualization
You can use gradio to see how estimation result maps to real satellite imagery map.
`gradio landuse_visualize.py`
&#8251; Currently only works for HUN Agricultural but uploaded it for reference.

```
  @article{kuckreja2023geochat,
          title={GeoChat: Grounded Large Vision-Language Model for Remote Sensing},
          author={Kuckreja, Kartik and Danish, Muhammad S. and Naseer, Muzammal and Das, Abhijit and Khan, Salman and Khan, Fahad S.},
          journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          year={2024}
  }
```
