
<br />
<p align="center">

  <h3 align="center">SpatiaLoc: Leveraging Multi-Level Spatial Enhanced Descriptors for Cross-Modal Localization</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2601.03579" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-yellow">
  </a>

<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/141d114c-704e-43e7-bb68-a7893b4bc36c" align="center" width="90%">
  <br>
    Cross-modal localization using text and point clouds enables robots to localize themselves via natural language descriptions, with applications in autonomous navigation and interaction between humans and robots. In this task, objects often recur across text and point clouds, making spatial relationships the most discriminative cues for localization. Based on this observation, we present SpatiaLoc, a framework utilizing a coarse-to-fine strategy that emphasizes spatial relationships at both the instance and global levels. In the coarse stage, we introduce a Bézier Enhanced Object Spatial Encoder (BEOSE) that models spatial relationships at the instance level using quadratic Bézier curves. Additionally, a Frequency Aware Encoder (FAE) generates global spatial representations in the frequency domain. In the fine stage, an Uncertainty Aware Gaussian Fine Localizer (UGFL) regresses 2D positions by modeling predictions as Gaussian distributions with an uncertainty-aware loss function. Extensive experiments on KITTI360Pose demonstrate that SpatiaLoc significantly outperforms existing state-of-the-art (SOTA) methods.
</p>
<br>

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
- [Train](#train)
- [Eval](#eval)
- [Test](#test)

## Installation

Create the environment using the following command.

```
git clone https://github.com/Anonymous09871745/SpatiaLoc

conda create -n Spatialoc python=3.10
conda activate Spatialoc

# Install the according versions of torch and torchvision
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```


## Data Preparation


We use the publicly available dataset KITTI360Pose. You can download the KITTI360Pose dataset from [here](https://cvg.cit.tum.de/webshare/g/text2pose/). 

For dataset details, kindly refer to [Text2Pos](https://arxiv.org/abs/2203.15125). 

The dataset folder should display as follow:

```html
data
└── KITTI360Pose
    └── k360_30-10_scG_pd10_pc4_spY_all
        ├── cells
        ├── direction
        ├── poses
        ├── street_centers
        └── visloc
```


## Model Zoo
The table below lists the pretrained weights in our method. These include the default text encoder and the 3D point cloud backbone. You can download them directly from the provided links.

| Component              | Model                                           | Download Link                                                                 |
|------------------------|-------------------------------------------------------|--------------------------------------------------------------------------------|
| **Text Backbone**       | Flan-T5                          | [Hugging Face](https://huggingface.co/google/flan-t5-large)                   |
| **Object Backbone** | PointNet  | [Google Drive](https://drive.google.com/file/d/1j2q67tfpVfIbJtC1gOWm7j8zNGhw5J9R/view) |



After completing the above steps, the basic directory structure should be like:

```
SpatiaLoc
 ├── checkpoints
      ├── coarse.pth
      ├── fine.pth
      └── pointnet_acc0.86_lr1_p256_model.pth
 ├── data
      └── KITTI360Pose
            └── k360_30-10_scG_pd10_pc4_spY_all
                ├── cells
                ├── direction
                ├── poses
                ├── street_centers
                └── visloc
 ├── dataloading
      └── .....
 ├── datapreparation
      └── .....
 ├── evalution
      └── .....
 ├── models
      └── .....
 ├── t5-large
      └── .....
 ├── training
      └── .....
```


## Train

After configuring the dependencies and preparing the dataset, use the following commands to train the coarse retrieval and fine localization, respectively.


**Coarse Retrieval**

```
python -m training.coarse  \
  --batch_size 64  \
  --coarse_embed_dim 256  \
  --shuffle  \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0001 \
  --lr_scheduler step \
  --lr_step 5 \
  --lr_gamma 0.5 \
  --temperature 0.05 \
  --ranking_loss CCL \
  --num_of_hidden_layer 3 \
  --alpha 2 \
  --hungging_model t5-large \
  --folder_name PATH_TO_COARSE
```


**Fine Localization**

```
python -m training.fine 
  --batch_size 32 \ 
  --fine_embed_dim 128 \ 
  --shuffle \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0003 \
  --fixed_embedding \
  --hungging_model t5-large \
  --regressor_cell all \
  --pmc_prob 0.5 \
  --folder_name PATH_TO_FINE \
```

## Eval

**Evaluation coarse retrieval only on val set**

```
python -m evaluation.coarse
	--base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --no_pc_augment \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
```

**Evaluation whole pipeline on val set**

```
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} \
```


## Test



**Test coarse retrieval only on test set**

```
python -m evaluation.coarse 
	--base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --use_test_set \
    --no_pc_augment \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
```


**Test whole pipeline on test set**

```
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --use_test_set \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```
