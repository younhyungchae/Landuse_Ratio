&#8251; This code is based on [GeoChat](https://github.com/mbzuai-oryx/GeoChat.git) repository. Refer that repository for more detailed informations.
# Landuse Ratio Estimation
This project estimates the landuse ratio of the single satellite image.

The model is trained with [SeasoNet](https://zenodo.org/records/5850307) dataset.

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

## Pretrained Model
Pretrained Model: [landuse_ratio_classification_geochat](https://huggingface.co/YounhyungChae/landuse_ratio_classification_geochat)

## Dataset
id : image id, not necessary
image: image path, required
converations: prompt and label
label: dictionary in the format of  {class: ratio_label}
###Dataset Example
```
{"id": "33UUQ_20180703T101029_48_789966_13_118461", "image": "33UUQ_20180703T101029_48_789966_13_118461.jpg", "conversations": [{"from": "human", "value": "<image>\nClassify the given image in one of the following classes. Classes: Residential, Agricultural, Commercial, Factory, Mining, Power station, Sports, Religious, Transportation, Water, Others. \nAnswer in one word or a short phrase.", "label": {"Residential": 0.07680555555555556, "Agricultural": 0.14381944444444444, "Commercial": 0, "Factory": 0, "Mining": 0, "Power station": 0, "Transportation": 0, "Sports": 0, "Religious": 0, "Water": 0, "Others": 0.26979166666666665}}]}
```

## Training
1. Download [Geochat Base Model](https://huggingface.co/MBZUAI/geochat-7B)
  ```
git lfs install
git clone https://huggingface.co/MBZUAI/geochat-7B
  ```
2. Remove Vision Tower from Geochat.
   
  &#8251; You must remove vision tower parameter as well as mapping dictionary in pytorch_model.bin.index.json
  
3. Train with DeepSpeed ZeRO-3: finetune_lora.sh
`bash scripts/finetune_lora.sh`

## Inference

```
  @article{kuckreja2023geochat,
          title={GeoChat: Grounded Large Vision-Language Model for Remote Sensing},
          author={Kuckreja, Kartik and Danish, Muhammad S. and Naseer, Muzammal and Das, Abhijit and Khan, Salman and Khan, Fahad S.},
          journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          year={2024}
  }
```
