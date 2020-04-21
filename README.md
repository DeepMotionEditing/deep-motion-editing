# Deep-motion-editing
This library provides fundamental and advanced functions to work with 3D character animation in deep learning using Pytorch. The code contains end-to-end modules, from reading and editing animation files to visualizing and rendering (using Blender) them.

The two main deep editing operations provided here, were proposed in [Skeleton-Aware Networks for Deep Motion Retargeting]() and [Unpaired Motion Style Transfer from Video to Animation](), which are published in SIGGRAPH 2020.

This library was written and is maintained by [Kfir Aberman](kfiraberman.github.io), [Peizhuo Li]() and [Yijia Weng]().


## Quick Start
We provide pretrained models and a few examples that enable one to retarget motion or transfer its style via

```bash
python test.py -model_path MODEL_PATH -input_A PATH_A -input_B PATH_B -edit_type TYPE
```

### Motion Retargeting
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
python test.py -model_path retargeting/models/pretrained_style_transfer.pth
-input_A style_transfer/examples/content_input -input_B style_transfer/examples/3D_style_input -edit_type style_transfer
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

Links, processing, comments.

#### Train

```bash
python retargeting/train.py
```

### Motion Style Transfer

#### Dataset

Links, processing, comments.

#### Train

```bash
python retargeting/train.py
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

Run `blender -P render.py` to get a ready to render blender GUI.

### Usage

#### Arguments

Due to blender's argparse system, the argument list should be separated from the python file with an extra '--', for example:

`blender -P render.py -- --arg1 [ARG1] --arg2 [ARG2]`

engine: "cycles" or "eevee". Please refer to `Render` section for more details.

render: 0 or 1. If set to 1, the scrip will start render before entering blender GUI. It is recommended to use render = 0 and render after starting blender GUI because you might need to adjust the camera. It's also recommended to use render = 1 with `blender -b -P render.py` to run blender in background without GUI.

The full parameters list can be displayed by:
`blender -P render.py -- -h`

#### Load bvh File (`load_bvh.py`)

You can use this file alone. To load `example.bvh` , run `blender -P load_bvh.py`. Please finish the preperation first.


> Currently it uses primitive_cone with 5 vertices for limbs.

> Note that Blender and bvh file have different xyz-coordinate systems. In bvh file, the "height" axis is y-axis while in blender it's z-axis. `load_bvh.py`  swaps the axis in the `BVH_file` class initialization funtion.

> Currently all the `End Sites` in bvh file are discarded, this is because of the out-side code used in `utils/`.

> After loading the bvh file, it's height is normalized to 10.





#### Material, Texture, Light and Camera (`scene.py`)

This file add a checkerboard floor, camera, a "sun" to the scene and apply a pure color material to character. The color and light is not very good now.

The floor doesn't work well now. It requires the animation in bvh file takes plan y=0 as floor, which is not true for all bvh files. But it works well with Mixamo's bvh.



## Rendering

Blender 2.80 provides 3 render engines: Eevee, Workbench, Cycles.

Eevee is a real-time render engine: fast, but it can't get very high quality result.

Workbench is not often used.

Cycles is an unbiased ray-tracing render engine: slow but provides photo-level rendering result. Cycles also supports CUDA and OpenGL acceleration.

Here is a comparison between the two engines. Cycles (right) provides more details (e.g., shadows).


<p float="left">
  <img src="blender_rendering/images/eevee.png" width="230" />
  <img src="blender_rendering/images/cycles.png" width="230" />
</p>



## Citation
If you use this modules for your research, please cite:
