# dynpoint: dynamic neural point for view synthesis

[Paper](https://arxiv.org/pdf/2310.18999.pdf) 

## Setup
To get started, please create the conda environment `py38` by running
```
conda env create -f environment.yml
```

## Template Dataset
We provide the two template scenarios from DAVIS, and you could find the link below:
[Dataset](https://drive.google.com/drive/folders/1r5TL_6PKa_wJspc6bz835Q9_D4unYmd5?usp=sharing) 

## Dataset Structure
You can adapt this code to any dataset with following structure:
```
-- dataset
    | camera
        | camera_xxxxx.npz
    | image
        | image_xxxxx.png
    | motion
        | mask_xxxxx.png
```

## Pretrained Weights
We provide the pretrained optical flow model and pretrained depth model, and you could find links below:

[Optical Flow (midas_cpkt.pth)](https://drive.google.com/drive/folders/1r5TL_6PKa_wJspc6bz835Q9_D4unYmd5?usp=sharing) 

[Depth (sintel.pth)](https://drive.google.com/drive/folders/1r5TL_6PKa_wJspc6bz835Q9_D4unYmd5?usp=sharing) 

Please place the pretrained weights files under a directory named 'pretrained_weights'.

## Training Step
### 1. Preprocessing dataset and estimating the scale factors of monocular depth.
```
python s0_train_depth.py
```

This command will create a directory named 'pair_file' that will contain pairwise dictionaries with the following naming convention:
```
-- dataset
　　| pair_file
        |gap_xx_cur_xxxxx.pt
```

Each pairwise dictionary in the 'pair_file' directory contains the following information: image, extrinsic matrix, intrinsic matrix, predicted optical flow, predicted depth, time ID, edge mask, corresponding mask, motion mask, and scale factors.

### 2. Estimating the scene flow based on preprocessed data.
```
bash ./experiments/davis/train_sequence.sh --track_id train_git --checkpoint exp_train --dataset davis_sequence
```

#### Error Map
![error_map.png](visualization/error_map.png)

### 3. Generating Pairwise scene flow file.
```
bash ./experiments/davis/val_sequence.sh --track_id train_git --dataset davis_sequence --resume checkpoints/exp_train/ckpt/07_0255.pth
```

This command will create a new directory named 'pair_file' that will contain pairwise dictionaries with the following naming convention:
```
-- dataset
　　| pair_file_final
        |gap_xx_cur_xxxxx.pt
```

Each pairwise dictionary in the 'pair_file' directory contains the following information: image, mask, scene flow, global point, and time ID.

### 4. Visualization.
```
python visualization.py
```
Aggregate colored point will be saved under directory 'test_img/points'.

#### Points
![points.png](visualization/points.png)


## To Be Done.
Adding rendering part.

## Acknowledgments
Our flow prediction code is modified from [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official).

Our depth prediction code is modified from [MiDaS](https://github.com/isl-org/MiDaS).




