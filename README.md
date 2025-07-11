# [Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models](https://openaccess.thecvf.com/content/WACV2025/html/Pesonen_Detecting_Wildfires_on_UAVs_with_Real-Time_Segmentation_Trained_by_Larger_WACV_2025_paper.html)

Boreal Forest Fire: UAV-collected Wildfire Detection and Smoke Segmentation Dataset is now available at [Fairdata](https://doi.org/10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9) and the Remixed Wildfire Smoke dataset on [Kaggle](https://www.kaggle.com/datasets/juliuspesonen/remixed-wildfire-smoke) or [Google drive](https://drive.google.com/file/d/1mbthNoeld-DnfrBxH8Ok9V1w5G52qC3P/view?usp=sharing), credits to [AI For Mankind and HPWREN](https://github.com/aiformankind/wildfire-smoke-dataset)! The paper was published at [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Pesonen_Detecting_Wildfires_on_UAVs_with_Real-Time_Segmentation_Trained_by_Larger_WACV_2025_paper.html).

![Real-time Segmentation Trained by Larger Teacher Models overview](sam_guided_seg.png)

To recreate results from the paper go to [Quick Start](#quick-start) after [Installation](#installation).

## Results and checkpoints

### [Inference video](https://www.youtube.com/shorts/xL4HofrKf2I)

Model comparison SAM supervised results (Table 2. Rows 5, 10, 15 and 20):

| Teacher Model | Student Model | Test mIoU |
|---------------|---------------|-----------|
| SAM           |               |   0.636   |
| SAM           | [PIDNet-S](https://drive.google.com/file/d/11QJKRjko9rIOpasyoo4Q9U6qzmRuAVg-/view?usp=drive_link)      |   0.594   | 
| SAM           | [PIDNet-M](https://drive.google.com/file/d/1ny_6mTnhPArqCnacGuidj9uc-tkTAokE/view?usp=drive_link)      |   0.606   | 
| SAM           | [PIDNet-L](https://drive.google.com/file/d/1ACFYUGqBHWo4wfxKJUdr9BwSjdTI11i5/view?usp=drive_link)      |   0.594   | 

Final SAM model with loss function ablation (Table 6. row 3.):

| Teacher Model | Student Model | Test mIoU |
|---------------|---------------|-----------|
| SAM | [PIDNet-S with loss ablation](https://drive.google.com/file/d/13wmJ_onh9p3eS0zTK_TyBR0OeRPKuh3P/view?usp=sharing) | 0.633 |

SAM supervised dataset separated tests (Table 4. row 5.):

| Teacher Model | Student Model | UAV mIoU | AI For Mankind mIoU |
|---------------|---------------|----------|---------------------|
| SAM | [PIDNet-S](https://drive.google.com/file/d/11QJKRjko9rIOpasyoo4Q9U6qzmRuAVg-/view?usp=drive_link) | 0.689 | 0.498 |


## Installation

The code has been developed in a Python 3.8 environment. For other Python versions package conflicts may occur. To create the environment with Conda (To install Conda follow the [official instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)):

`git clone https://gitlab.com/fgi_nls/public/wildfire-real-time-segmentation.git`

`cd wildfire-real-time-segmentation`

`conda create -n wildfire_seg python=3.8`

`conda activate wildfire_seg`

Optional step required only if you want to generate masks with SAM:

`pip install git+https://github.com/facebookresearch/segment-anything.git`

Required for everything:

`pip install -r requirements.txt`

On Windows systems to enable CUDA support follow the [official Pytorch instructions](https://pytorch.org/get-started/locally/).

## Quick start

Before running the python scripts follow the installation instructions in [Installation](#installation). Download the reordered [AI For Mankind Data](https://drive.google.com/file/d/1mbthNoeld-DnfrBxH8Ok9V1w5G52qC3P/view?usp=sharing) and the Subset-C from the [Boreal Forest Fire data](https://doi.org/10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9) and extract them to the same directory.

All code can be run on a CPU with the argument `--device cpu`.

![BFF dataset examples](data_vis/example_imgs.png)
|:--:| 
| *Examples from the Boreal Forest Fire dataset with SAM generated masks.* |


### Recreate the results with pretrained models

Download the model checkpoints from [Results and checkpoints](#results-and-checkpoints) and place them in the checkpoints folder.

Then to recreate individual model results:

`python eval.py --single-model [model-checkpoint-path] --model-size [s,m,l]`

To evaluate only the SAM pseudo-labels:

`python eval.py --eval-sam`

### Generate the wildfire masks with SAM

#### Requirements: 

From the SAM repository: https://github.com/facebookresearch/segment-anything download the trained model checkpoints and place them in the folder 'checkpoints'. For the study, we have used the ViT-H model.

#### Generate masks:

`python generate_pseudo_labels.py --model vit_h --dir data --mode test --output-dir pseudo_labels`

To generate masks for custom data follow the data organisation described in [Model training](#model-training).

### Model training

To train:

`python train.py`

For training with custom data, organise the data according to the default dataset:

```
├── custom_data
| ├── images
| | ├── train
| | ├── valid
| | ├── test
| ├── labels
| | ├── train
| | ├── valid
| | ├── test
| ├── sam_masks
| | ├── train
| | ├── valid
| | ├── test
```

`python train.py --data-dir custom_data`

To generate the pseudo-labels with SAM refer to [Generate masks](#generate-masks).

### Visualise outputs

To generate model output visualisations:

`python infer.py --input-image [image path]`


## Citation

```
@InProceedings{Pesonen_2025_WACV,
    author    = {Pesonen, Julius and Hakala, Teemu and Karjalainen, V\"ain\"o and Koivum\"aki, Niko and Markelin, Lauri and Raita-Hakola, Anna-Maria and Suomalainen, Juha and P\"ol\"onen, Ilkka and Honkavaara, Eija},
    title     = {Detecting Wildfires on UAVs with Real-Time Segmentation Trained by Larger Teacher Models},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {5166-5176}
}

@misc{Pesonen_2025_fairdata,
author = {Pesonen, Julius and Raita-Hakola, Anna-Maria and Joutsalainen, Jukka and Hakala, Teemu and Akhtar, Waleed and Karjalainen, Väinö and Koivumäki, Niko and Markelin, Lauri and Suomalainen, Juha and de Oliveira, Raquel Alves and others},
title = {Boreal Forest Fire: UAV-collected Wildfire Detection and Smoke Segmentation Dataset},
howpublished = {\url{https://doi.org/10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9}},
month = {2},
year = {2025},
note = {National Land Survey of Finland, FGI Dept. of Remote sensing and photogrammetry}
}
```

## Acknowledgement

* The inference model and its training implementation is based on and largely built on top of the code of [PIDNet](https://github.com/XuJiacong/PIDNet)
* The introduced training method relies on supervision from [Segment Anything](https://github.com/facebookresearch/segment-anything)