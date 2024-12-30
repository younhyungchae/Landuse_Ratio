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

## Training
1. Download [Geochat Base Model](https://huggingface.co/MBZUAI/geochat-7B)
  ```
git lfs install
git clone https://huggingface.co/MBZUAI/geochat-7B
  ```
2. Remove Vision Tower from Geochat. (&#8251; You must remove vision tower parameter as well as mapping dictionary in pytorch_model.bin.index.json)
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
