# Deep-motion-editing

![Python](https://img.shields.io/badge/Python->=3.7-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.5.0-Red?logo=pytorch)
![Blender](https://img.shields.io/badge/Blender-%3E=2.8-Orange?logo=blender)

This library provides fundamental and advanced functions to work with 3D character animation in deep learning with Pytorch. The code contains end-to-end modules, from reading and editing animation files to visualizing and rendering (using Blender) them.

The main deep editing operations provided here, motion retargeting and motion style transfer, are based on two works published in SIGGRAPH 2020:

**Skeleton-Aware Networks for Deep Motion Retargeting**: [Project](https://deepmotionediting.github.io/retargeting) | [Paper](https://arxiv.org/abs/2005.05732) |
[Video](https://www.youtube.com/watch?v=ym8Tnmiz5N8)

<img src="images/retargeting_teaser.gif" align="center"> <br>

**Unpaired Motion Style Transfer from Video to Animation**: [Project](https://deepmotionediting.github.io/style_transfer) | [Paper](https://arxiv.org/abs/2005.05751) |
[Video](https://www.youtube.com/watch?v=m04zuBSdGrc)

<img src="images/style_transfer_teaser.gif" align="center"> <br>


This library is written and maintained by [Kfir Aberman](https://kfiraberman.github.io), [Peizhuo Li](https://peizhuoli.github.io/) and [Yijia Weng](https://halfsummer11.github.io/). The library is still under development.

## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Quick Start
We provide pretrained models together with demo examples using animation files specified in bvh format.

<!--```bash
python test.py -model_path MODEL_PATH -input_A PATH_A -input_B PATH_B -edit_type TYPE
```-->

### Motion Retargeting
<!-- `TYPE = retargeting`  
`PATH_A` - motion input  
`PATH_B` - skeleton input -->

Download and extract the test dataset from [Google Drive](https://docs.google.com/uc?export=download&id=1_849LvuT3WBEHktBT97P2oMBzeJz7-UP) or [Baidu Disk](https://pan.baidu.com/s/1z1cQiqLUgjfxlWoajIPr0g) (ye1q). Then place the `Mixamo` directory within `retargeting/datasets`.

To generate the demo examples with the pretrained model, run

```bash
cd retargeting
sh demo.sh
```

The results will be saved in `retargeting/examples`.

To reconstruct the quantitative result with the pretrained model, run

```bash
cd retargeting
python test.py
```

The retargeted demo results, that consists both intra-structual retargeting and cross-structural retargeting, will be saved in `retargeting/pretrained/results`.

<!-- The system support both in Intra-Structural retargeting:
```bash
python test.py -model_path retargeting/models/pretrained_retargeting.pth -input_A retargeting/examples/IS_motion_input -input_B retargeting/examples/IS_skeleton_input -edit_type retargeting
```
(demo result GIF: input_motion, input_skeleton, output)

and Cross-structural retargeting:
```bash
python test.py -model_path retargeting/models/pretrained_retargeting.pth -input_A retargeting/examples/CS_motion_input -input_B retargeting/examples/CS_skeleton_input -edit_type retargeting
```

(demo result GIF: input_motion, input_skeleton, output) -->

### Motion Style Transfer

<!-- `TYPE = style_transfer`  
`PATH_A` - content motion input  
`PATH_B` - style motion input  

The system support both in style from 3D MoCap data:

```bash
python test.py -model_path retargeting/models/pretrained_style_transfer.pth -input_A style_transfer/examples/content_input -input_B style_transfer/examples/3D_style_input -edit_type style_transfer
```

(demo result GIF: input_content, input_style, output)

and in style from 2D key-points (extracted from video):

```bash
python test.py -model_path retargeting/models/pretrained_style_transfer.pth -input_A style_transfer/examples/content_input -input_B style_transfer/examples/2D_style_input -edit_type style_transfer
```
(demo result GIF: input_content, input_style_video, output) -->
To receive the demo examples, simply run
```bash
sh style_transfer/demo.sh
```
The results will be saved in `style_transfer/demo_results`,
where each folder contains the raw output `raw.bvh` and the output after footskate clean-up `fixed.bvh`.

## Train from scratch
We provide instructions for retraining our models

### Motion Retargeting

#### Dataset

We use Mixamo dataset to train our model. You can download our preprocessed data from [Google Drive](https://drive.google.com/file/d/1BYH2t5XMGWwnu5coftehU0rTXupQvFLg/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1PM0maLNCJxBsR9RIv18cTA)(4rgv). Then place the `Mixamo` directory within `retargeting/datasets`.

Otherwise, if you want to download Mixamo dataset or use your own dataset, please follow the instructions below. Unless specifically mentioned, all script should be run in `retargeting` directory.

+ To download Mixamo on your own, you can refer to [this](https://github.com/ChrisWu1997/2D-Motion-Retargeting/blob/master/dataset/Guide%20For%20Downloading%20Mixamo%20Data.md) good tutorial. You will need to download as fbx file (skin is not required) and make a subdirectory for each character in `retargeting/datasets/Mixamo`. In our original implementation we download 60fps fbx files and downsample them into 30fps. Since we use an unpaired way in training, it is recommended to divide all motions into two equal size sets for each group and equal size sets for each character in each group. If you use your own data, you need to make sure that your dataset consists of bvh files with same t-pose. You should also put your dataset in subdirectories of `retargeting/datasets/Mixamo`.

+ Enter `retargeting/datasets` directory and run `blender -b -P fbx2bvh.py` to convert fbx files to bvh files. If you already have bvh file as dataset, please skil this step.

+ In our original implementation, we manually split three joints for skeletons in `group A`. If you want to follow our routine, run `python datasets/split_joint.py`. This step is optional.

+ Run `python datasets/preprocess.py` to simplify the skeleton by removing some less interesting joints, e.g. fingers and convert bvh files into npy files. If you use your own data, you'll need to define simplified structure in `retargeting/datasets/bvh_parser.py`. This information currently is hard-coded in the code. See the comment in source file for more details. There are four steps to make your own dataset work.

+ Training and testing character are hard-coded in `retargeting/datasets/__init__.py`. You'll need to modify it if you want to use your own dataset.

#### Train
After preparing dataset, simply run 
```bash
cd retargeting
python train.py --save_dir=./training/
```

It will use default hyper-parameters to train the model and save trained model in `retargeting/training` directory. More options are available in `retargeting/option_parser.py`. You can use tensorboard to monitor the training progress by running

```bash
tensorboard --logdir=./retargeting/training/logs/
```

### Motion Style Transfer

#### Dataset

+ Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1C-_iZJj-PSUWZwh25yAsQe1tLpPm9EZ5?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1LtZaX7bQ6kz8TrDWd4FxWA) (zzck). The dataset consists of two parts: one is the taken from the motion style transfer dataset proposed by [Xia et al.](http://faculty.cs.tamu.edu/jchai/projects/SIG15/style-final.pdf) and the other is our BFA dataset, where both parts contain .bvh files retargeted to the standard skeleton of [CMU mocap dataset](http://mocap.cs.cmu.edu/).
+ Extract the .zip files into `style_transfer/data`


+ Pre-process data for training:

  ```bash
  cd style_transfer/data_proc
  sh gen_dataset.sh
  ```

  This will produce `xia.npz`, `bfa.npz` in `style_transfer/data`.

#### Train

After downloading the dataset simply run
```bash
python style_transfer/train.py
```

#### Style from videos

To run our models in test time with your own videos, you first need to use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract the 2D joint positions from the video, then use the resulting JSON files as described in the demo examples.

## Blender Visualization
We provide a simple wrapper of blender's python API (2.80) for rendering 3D animations.

### Prerequisites

> The Blender releases distributed from blender.org include a complete Python installation across all platforms, which means that any extensions you have installed in your systems Python won’t appear in Blender.

To use external python libraries, you can install new packages directly to Blender's python distribution. Alternatively, you can change the default blender python interpreter by:

1. Remove the built-in python directory: `[blender_path]/2.80/python`.

2. Make a symbolic link or simply copy a python interpreter at [blender_path]/2.80/python. E.g. `ln -s ~/anaconda3/envs/env_name [blender_path]/2.80/python`

This interpreter should be python 3.7.x version and contains at least: numpy, scipy.

### Usage

#### Arguments

Due to blender's argparse system, the argument list should be separated from the python file with an extra '--', for example:

`blender -P render.py -- --arg1 [ARG1] --arg2 [ARG2]`

engine: "cycles" or "eevee". Please refer to `Render` section for more details.

render: 0 or 1. If set to 1, the data will be rendered outside blender's GUI. It is recommended to use render = 0 in case you need to manually adjust the camera.

The full parameters list can be displayed by:
`blender -P render.py -- -h`

#### Load bvh File (`load_bvh.py`)

To load `example.bvh`, run `blender -P load_bvh.py`. Please finish the preparation first.


> Note that currently it uses primitive_cone with 5 vertices for limbs.

> Note that Blender and bvh file have different xyz-coordinate systems. In bvh file, the "height" axis is y-axis while in blender it's z-axis. `load_bvh.py`  swaps the axis in the `BVH_file` class initialization funtion.

> Currently all the `End Sites` in bvh file are discarded, this is because of the out-side code used in `utils/`.

> After loading the bvh file, it's height is normalized to 10.



#### Material, Texture, Light and Camera (`scene.py`)

This file enables to add a checkerboard floor, camera, a "sun" to the scene and to apply a basic color material to character.

The floor is placed at y=0, and should be corrected manually in case that it is needed (depends on the character parametes in the bvh file).


## Rendering

We support 2 render engines provided in Blender 2.80: Eevee and Cycles, where the trade-off is between speed and quality.

Eevee (left) is a fast, real-time, render engine provides limited quality, while Cycles (right) is a slower, unbiased, ray-tracing render engine provides photo-level rendering result. Cycles also supports CUDA and OpenGL acceleration.


<p float="left">
  <img src="blender_rendering/images/eevee.png" width="300" />
  <img src="blender_rendering/images/cycles.png" width="300" />
</p>

## Skinning

### Automatic Skinning

We provide a blender script that applies "skinning" to the output skeletons. You first need to download the fbx file which corresponds to the targeted character (for example, "[mousey](https://www.mixamo.com/#/?page=1&query=mousey&type=Character)"). Then, you can get a skinned animation by simply run

```sh
blender -P blender_rendering/skinning.py -- --bvh_file [bvh file path] --fbx_file [fbx file path]
```

Note that the script might not work well for all the fbx and bvh files. If it fails, you can try to tweak the script or follow the manual skinning guideline below.


### Manual Skinning

Here we provide a "quick and dirty" guideline for how to apply skin to the resulting bvh files, with blender:

- Download the fbx file that corresponds to the retargeted character (for example, "[mousey](https://www.mixamo.com/#/?page=1&query=mousey&type=Character)")
- Import the fbx file to blender (uncheck the "import animation" option)
- Merge meshes - select all the parts and merge them (ctrl+J)
- Import the retargeted bvh file
- Click "context" (menu bar) -> "Rest Position" (under sekeleton)
- Manually align the mesh and the skeleton (rotation + translation)
- Select the skeleton and the mesh (the skeleton object should be highlighted)
- Click Object -> Parent -> with automatic weights (or Ctrl+P)

Now the skeleton and the skin are bound and the animation can be rendered.


## Acknowledgments
The code in the utils directory is mostly taken from [Holden et al. [2016]](http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing).  
In addition, part of the MoCap dataset is taken from [Adobe Mixamo](https://www.mixamo.com/) and from the work of [Xia et al.](http://faculty.cs.tamu.edu/jchai/projects/SIG15/style-final.pdf).

## Citation
If you use this code for your research, please cite our papers:
```bibtex
@article{aberman2020skeleton,
  author = {Aberman, Kfir and Li, Peizhuo and Sorkine-Hornung Olga and Lischinski, Dani and Cohen-Or, Daniel and Chen, Baoquan},
  title = {Skeleton-Aware Networks for Deep Motion Retargeting},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {39},
  number = {4},
  pages = {62},
  year = {2020},
  publisher = {ACM}
}
```
and
```bibtex
@article{aberman2020unpaired,
  author = {Aberman, Kfir and Weng, Yijia and Lischinski, Dani and Cohen-Or, Daniel and Chen, Baoquan},
  title = {Unpaired Motion Style Transfer from Video to Animation},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {39},
  number = {4},
  pages = {64},
  year = {2020},
  publisher = {ACM}
}
```
