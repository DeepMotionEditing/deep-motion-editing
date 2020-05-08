## Motion Style Transfer

### Produce demo results

We provide a pretrained model and several sample test inputs, from which the main results shown in the video and the paper can be reproduced. 

+ To reproduce the style transfer results in the video, run

  ```bash
  bash style_transfer/test.sh
  ```

  The results can be found in `style_transfer/demo_results`. They include (in the order of their appearance in the video)
  
  + `demo_3d_1, demo_3d_2`: demo results using 3D motion (in .bvh format) as style input
  + `demo_video_1, demo_video_2`: demo results using 2D motion (keypoints in .json format, extracted by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)) as style input
  + `comp_3d_1, comp_3d_2`: comparison with [Holden 2016](http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing) using 3D motion as style input
  + `comp_video_1, comp_video_2`: comparison with [Holden 2016](http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing) using 2D motion as style input
  
  Each folder contains the raw output `raw.bvh` and the output after footskate clean-up `fixed.bvh`.
  
+ To reproduce the latent space figures in the paper, run

  ```bash
  bash style_transfer/plot_demo_figures.sh
  ```

  The figures can be found in `style_transfer/demo_results/figures`. They include:

  + `style3d_adain_tsne`: Fig. 5 (b).

  + `joint_embedding_adain_tsne`: Fig. 6.

  + `content_by_style`, `content_by_phase`: Fig. 8 (a)(b).

  + `style3d_code_tsne`: Fig. 10. (right).

    

### Train from scratch

#### Prepare Data

+ Download Xia's dataset and BFA dataset via [link](http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing). The datasets contain .bvh files retargeted to the standard skeleton in [CMU mocap dataset](http://mocap.cs.cmu.edu/) . 

+ Pre-process data for training:

  ```bash
  mv {mocap_xia.zip,mocap_bfa.zip} style_transfer/data
  cd style_transfer/data_proc
  bash gen_dataset.sh
  ```

  This will produce `xia.npz`, `bfa.npz` in `style_transfer/data`.

#### Train

- Run

  ```bash
  bash style_transfer/train.sh
  ```

  
