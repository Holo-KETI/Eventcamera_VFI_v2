# KETI_DVS_VFI_V2

## Getting Started
### Installation
For environment setup and installation instructions, please refer to the TimeLens-XL GitHub repository(https://github.com/OpenImagingLab/TimeLens-XL).
Our code is based on their implementation.

### Download Pretrained Model & bsergb dataset
1. Download Link : [x5 interpolation weights](https://drive.google.com/file/d/1E8n7H0KZKRbn6VSLvBhTPwJIW9aYzPSS/view?usp=sharing)
```bash
$ mkdir weights
$ cd weights
```
Place downloaded weights under "./weights" folder.

2. Download Link: [Public Dataset_BSERGB](https://drive.google.com/file/d/1DPY0G1sr2TfP_Pt0rZpSgmCNIwbwtbNe/view?usp=sharing)

 ```bash
$ cd dataset
```
Place downloaded dataset under "./dataset" folder.

Modify the dataset path in params/Paths/BSERGB.py to point to your ```Local PATH```.

3. Download Link: [Dists pytorch weight](https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/weights.pt)
```bash
$ cd ./losses/DISTS/DISTS_pytorch
```
Place downloaded weights.pt under "./losses/DISTS/DISTS_pytorch" folder.

Modify the dataset path in ./losses/DISTS/DISTS_pytorch/DISTS_pt.py (63 row) to point to your ```Local PATH```.

## Inference model (x5 interpolation)
```bash
$ python run_network.py --param_name traintest_BSERGB_x5AdamwithLPIPS_vali --model_name Expv8_large --model_pretrained <pretrained_model_weight path> --skip_training
```

## Block code inference 
```bash
python block_all_pipeline.py
```
test set : Download Link: [block ver. test set](https://drive.google.com/file/d/1J8QBEyvSqWYVFNizdjSTg67KirDk-BX0/view?usp=sharing) <--- params.validation_config.data_paths

## Metric (psnr)
```bash
python metric.py --res_root <output_folder_path> --gt_root ./dataset/bs_ergb/1_TEST
```



