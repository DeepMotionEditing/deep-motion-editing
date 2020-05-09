# Deep-motion-editing
This library provides fundamental and advanced functions to work with 3D character animation in deep learning using Pytorch. The code contains end-to-end modules, from reading and editing animation files to visualizing and rendering (using Blender) them.

The two main deep editing operations provided here, were proposed in [Skeleton-Aware Networks for Deep Motion Retargeting]() and [Unpaired Motion Style Transfer from Video to Animation](), which are published in SIGGRAPH 2020.

This library was written and is maintained by [Kfir Aberman](kfiraberman.github.io), [Peizhuo Li]() and [Yijia Weng]().


## Quick Start
We provide pretrained models and a few examples that enable one to retarget motion or transfer its style via

```bash
python test.py -model_path MODEL_PATH -input_A PATH_A -input_B PATH_B -edit_type TYPE
```

### Motion Retargeting (Not yet supported)
`TYPE = retargeting`  
`PATH_A` - motion input  
`PATH_B` - skeleton input



The system support both in Intra-Structural retargeting:
```bash
python test.py -model_path retargeting/models/pretrained_retargeting.pth -input_A retargeting/examples/IS_motion_input -input_B retargeting/examples/IS_skeleton_input -edit_type retargeting
```
(demo result GIF: input_motion, input_skeleton, output)

and Cross-structural retargeting:
```bash
python test.py -model_path retargeting/models/pretrained_retargeting.pth -input_A retargeting/examples/CS_motion_input -input_B retargeting/examples/CS_skeleton_input -edit_type retargeting
```

(demo result GIF: input_motion, input_skeleton, output)

### Motion Style Transfer
`TYPE = style_transfer`  
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
(demo result GIF: input_content, input_style_video, output)

## Train from scratch
We provide instructions for retraining our models

### Motion Retargeting

#### Dataset

Here are the links to test data set: [Google Drive](https://docs.google.com/uc?export=download&id=1_849LvuT3WBEHktBT97P2oMBzeJz7-UP), [Baidu Disk](https://pan.baidu.com/s/1z1cQiqLUgjfxlWoajIPr0g) (ye1q). 

#### Train

Coming soon...

#### Test and Evaluation

Extract the test set from the download file and put the `Mixamo` directory in `retargeting/datasets`. Then run the following commands.

```bash
cd retargeting
python test.py
```

The retarget result is in `retargeting/pretrained/results` now and you can get the quantitative result. For intra structure retargeting, we test between all paires betweent the four test characters in `group A`. For cross structure, we test from `BigVegas` to all four test characters in `group A`.

### Motion Style Transfer

#### Dataset

Links (Xia and BFA), processing, comments.

#### Train

```bash
python style_transfer/train.py
```

#### Style from videos

To run our models in test time with your own videos, you first need to use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract the 2D joint positions from the video, then use the resulting JSON files as described in the demo examples.

## Blender Visualization
We provide a simple wrapper of blender's python API (2.80) for rendering 3D animations.

### Prerequisites

> The Blender releases distributed from blender.org include a complete Python installation across all platforms, which means that any extensions you have installed in your systems Python won’t appear in Blender.

To use external python libraries, you need to change the default blender python interpreter by:

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



## Citation
If you use this modules for your research, please cite:
