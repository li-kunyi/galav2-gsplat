# Occam's LGS: An Efficient Approach for Language Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-2412.01807-b31b1b.svg)](https://arxiv.org/abs/2412.01807)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://insait-institute.github.io/OccamLGS/)

This is the official implementation of "Occam's LGS: An Efficient Approach for Language Gaussian Splatting".

## Overview

Occam's LGS is a simple, training-free approach for Language-guided 3D Gaussian Splatting that achieves state-of-the-art results with a 100x speed improvement. Our method:

- 🎯 Lifts 2D language features to 3D Gaussian Splats without complex modules or training
- 🚀 Provides 100x faster optimization compared to existing methods  
- 🧩 Works with any feature dimension without compression
- 🎨 Enables easy scene manipulation and object insertion

## Installation Guide

### System Requirements
We use the following setting to run OccamLGS:

- NVIDIA GPU with CUDA support
- PyTorch 2.2.2
- Python 3.10
- GCC 11.4.0

### Clone Repository
```bash
git clone git@github.com:JoannaCCJH/occamlgs.git --recursive
```

### Environment Setup
```bash
micromamba create -n occamlgs python=3.10
micromamba activate occamlgs
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```

### Project Dependencies
```bash
pip install -r requirements.txt
micromamba install -c conda-forge gxx=11.4.0
```

### Submodules
```bash
# pip install -e submodules/gsplat[dev]
pip install -e submodules/gsplat --no-build-isolation
pip install -e submodules/simple-knn --no-build-isolation
```

## Dataset Preparation
### Input Dataset
The dataset follows a structured format where each 3D scene is organized as follows:
```
lerf_ovs/
└── scene_name/           # Name of the specific scene (e.g., teatime)
    ├── distorted/        
    ├── images/           # Contains the original, unprocessed scene images
    ├── language_features/ # Pre-extracted language embeddings
    │   ├── frame_00001_f.npy
    │   └── frame_00001_s.npy
    │   ├── ...
    ├── sparse/0/      
    │   ├── test.txt     # Testing image list
    │   ├── cameras.bin 
    │   ├── images.bin
    │   └── points3D.bin 
    ├── stereo/         
```
Notes:
- Language features are pre-extracted and stored as 512-dimensional vectors
- For detailed information about feature levels and language feature extraction methodology, please refer to the [LangSplat repository](https://github.com/minghanqin/LangSplat). 

### Output Directory Structure
The pre-trained RGB model outputs are organized as follows:
```
output/
└── dataset_name/
    └── scene_name/
        ├── point_cloud/
        │   └── iteration_30000/
        │       └── point_cloud.ply      # Point cloud at 30K iterations
        ├── cameras.json                 
        ├── cfg_args                     
        ├── chkpnt30000.pth             # Model checkpoint at 30K iterations
        └── input.ply                    

```
After running the `gaussian_feature_extractor.py` for three levels of features, three additional checkpoint files are added:

```
output/
└── dataset_name/
    └── scene_name/
        ├── point_cloud/
        │   └── iteration_30000/
        │       └── point_cloud.ply      # Point cloud at 30K iterations
        ├── cameras.json                
        ├── cfg_args                    
        ├── chkpnt30000.pth             # RGB model checkpoint
        ├── input.ply                   
        ├── chkpnt30000_langfeat_1.pth  # Language features level 1
        ├── chkpnt30000_langfeat_2.pth  # Language features level 2
        └── chkpnt30000_langfeat_3.pth  # Language features level 3

```

Note:  The script `gaussian_feature_extractor.py` generates three new semantic checkpoints, each containing a different level of language features while maintaining the same RGB model weights from the original checkpoint.

## Usage


### Prerequisites

-  A pre-trained RGB Gaussian model (use `train.py` and `render.py` commands below to train a model on your scene using gsplat renderer)
- `test.txt` file in `scene_name/sparse/0/` defining test set


#### 1. Train and Render RGB Gaussian Model
```bash
# Train gaussian model
python train.py -s $DATA_SOURCE_PATH -m $MODEL_OUTPUT_PATH --iterations 30000

# Render trained model
python render.py -m $MODEL_OUTPUT_PATH --iteration 30000
```

#### 2. Feature Extraction and Visualization
```bash
#  gaussian feature vectors
python gaussian_feature_extractor.py -m $MODEL_OUTPUT_PATH --iteration 30000 --eval --feature_level 1

# Render feature maps
python feature_map_renderer.py -m $MODEL_OUTPUT_PATH --iteration 30000 --eval --feature_level 1
```
### Example Pipeline
Check `run_lerf.sh` for a complete example using the "teatime" scene from LERF_OVS dataset and `run_3DOVS.sh` for a complete example using the "bench" scene from 3D-OVS dataset.

## Evaluation
### LERF
We follow the evaluation methodology established by LangSplat for our LERF assessments. For detailed information about the evaluation metrics and procedures, please refer to the LangSplat methodology.

### 3DOVS
Here is the instructions on how to evaluate 3DOVS Dataset.
1. Configure Parameters: Open `eval_3DOVS.sh` and adjust the following:
    - `DATASET_NAME`: Set to your 3DOVS dataset split (e.g., "bench")
    - `GT_FOLDER`: Path to your preprocessed 3DOVS data
    - `FEAT_FOLDER_NAME`: Name of your model's feature output folder
2. Run the evaluation script
```bash
sh eval_3DOVS.sh
```
3. View Results: Evaluation metrics and visualizations will be saved to the `/eval_results` directory

**Configuration Options**

The evaluation script supports several parameters:

- `--stability_thresh`: Threshold for stability analysis (default: 0.4)
- `--min_mask_size`: Minimum valid mask size (default: 0.005)
- `--max_mask_size`: Maximum valid mask size (default: 0.9)

For detailed information about our evaluation methodology, please refer to the supplementary materials in our paper.


## TODO
- [x] Training and rendering code released
- [x] GSplat rasterizer code released
- [x] Evaluation code to be released
- [ ] Corrected room scene labels to be released
- [ ] Autoencoder for any-dimensional feature to be released

## Acknowledgement
Our code is built on [LangSplat](https://github.com/minghanqin/LangSplat), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), and [gsplat](https://github.com/nerfstudio-project/gsplat). We gratefully appreciate their open source contribution!

## BibTeX

```bibtex
@article{cheng2024occamslgssimpleapproach,
 title={Occam's LGS: A Simple Approach for Language Gaussian Splatting}, 
 author={Jiahuan Cheng and Jan-Nico Zaech and Luc Van Gool and Danda Pani Paudel},
 year={2024},
 eprint={2412.01807}
}
