import os
import sys
import torch
from os.path import join as pjoin
from py_utils import ensure_dirs
import shutil
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)


class Config:

    for_try = False  # set to True only if you want to quickly check if all parts (latent space visualization, result output, etc.) function correctly 

    # Save & Visualization
    name = 'pretrained'     # name of the experiment, for training from scratch please use a different name

    cuda_id = 0

    # hyyyper params

    use_rotloss = True
    use_newdecoder = True

    # data paths
    data_dir = pjoin(BASEPATH, 'data')
    expr_dir = BASEPATH
    data_filename = "xia.npz"   # change to 'bfa.npz' for training on bfa data
    data_path = pjoin(data_dir, data_filename)
    extra_data_dir = pjoin(data_dir, data_filename.split('.')[-2].split('/')[-1] + "_norms")

    # model paths
    main_dir = None
    model_dir = None
    tb_dir = None
    info_dir = None
    output_dir = None

    vis_freq = 100
    log_freq = 100
    save_freq = 50000
    mt_save_iter = 50000       # How often do you want to save output images during training
    mt_display_iter = 5000       # How often do you want to display output images during training
    mt_batch_n = 1  # number of batches to save in training

    # optimization options
    max_iter = 300000              # maximum number of training iterations
    weight_decay = 0.0001          # weight decay
    lr_gen = 0.0001                # learning rate for the generator
    lr_dis = 0.0001                # learning rate for the discriminator
    weight_init = 'kaiming'                 # initialization [gaussian/kaiming/xavier/orthogonal]
    lr_policy = None

    # Training
    batch_size = 128

    # Testing
    test_batch_n = 56  # number of test clips

    if for_try:
        vis_freq = 1
        log_freq = 1
        save_freq = 5
        # logger options
        mt_save_iter = 2       # How often do you want to save output images during training
        mt_display_iter = 3       # How often do you want to display output images during training
        mt_batch_n = 1  # number of batches to save in training

        # max_iter = 10              # maximum number of training iterations
        batch_size = 16

    # dataset
    dataset_norm_config = {  # specify the prefix of mean/std
        "train":
            {"content": None, "style3d": None, "style2d": None},  # will be named automatically as "train_content", etc.
        "test":
            {"content": "train", "style3d": "train", "style2d": "train"},
        "trainfull":
            {"content": "train", "style3d": "train", "style2d": "train"}
    }

    # input: T * 64
    rot_channels = 128  # added one more y-axis rotation
    pos3d_channels = 64  # changed to be the same as rfree
    proj_channels = 42

    num_channel = rot_channels
    num_style_joints = 21

    style_channel_2d = proj_channels
    style_channel_3d = pos3d_channels

    """
    encoder for class
    [down_n] stride=[enc_cl_stride], dim=[enc_cl_channels] convs, 
    followed by [enc_cl_global_pool]

    """
    enc_cl_down_n = 2  # 64 -> 32 -> 16 -> 8 -> 4
    enc_cl_channels = [0, 96, 144]
    enc_cl_kernel_size = 8
    enc_cl_stride = 2

    """
    encoder for content
    [down_n] stride=[enc_co_stride], dim=[enc_co_channels] convs (with IN)
    followed by [enc_co_resblks] resblks with IN
    """
    enc_co_down_n = 1  # 64 -> 32 -> 16 -> 8
    enc_co_channels = [num_channel, 144]
    enc_co_kernel_size = 8
    enc_co_stride = 2
    enc_co_resblks = 1


    """
    mlp
    map from class output [enc_cl_channels[-1] * 1]
    to AdaIN params (dim calculated at runtime)
    """
    mlp_dims = [enc_cl_channels[-1], 192, 256]

    """
    decoder
    [dec_resblks] resblks with AdaIN
    [dec_up_n] Upsampling followed by stride=[dec_stride] convs
    """

    dec_bt_channel = 144
    dec_resblks = enc_co_resblks
    dec_channels = enc_co_channels.copy()
    dec_channels.reverse()
    dec_channels[-1] = 31 * 4  # Let it output rotations only
    dec_up_n = enc_co_down_n
    dec_kernel_size = 8
    dec_stride = 1

    """
    discriminator
    1) conv w/o acti or norm, keeps dims
    2) [disc_down_n] *
            (ActiFirstResBlk(channel[i], channel[i])
            + ActiFirstResBlk(channel[i], channel[i + 1])
            + AvgPool(pool_size, pool_stride))
    3) 2 ActiFirstResBlks that keep dims(channel[-1])
    4) conv, [channel[-1] -> num_classes]

    """
    disc_channels = [pos3d_channels, 96, 144]
    disc_down_n = 2  # 64 -> 32 -> 16 -> 8 -> 4
    disc_kernel_size = 6
    disc_stride = 1
    disc_pool_size = 3
    disc_pool_stride = 2

    num_classes = 8         # set to 16 for training on bfa data

    gan_w = 1
    rec_w = 1
    rrec_w = 1
    feat_w = 0.5
    qt_w = 0.1
    joint_w = 0.3
    triplet_w = 0.3
    triplet_margin = 5
    twist_w = 1
    twist_alpha = 100
    trans_weight = 0.5

    device = None
    gpus = 1

    def initialize(self, args=None, save=True):

        if hasattr(args, 'name') and args.name is not None:
            print("args.name= ", args.name)
            self.name = args.name

        if hasattr(args, 'batch_size') and args.name is not None:
            self.batch_size = args.batch_size

        self.main_dir = os.path.join(self.expr_dir, self.name)
        self.model_dir = os.path.join(self.main_dir, "pth")
        self.tb_dir = os.path.join(self.main_dir, "log")
        self.info_dir = os.path.join(self.main_dir, "info")
        self.output_dir = os.path.join(self.main_dir, "output")

        ensure_dirs([self.main_dir, self.model_dir, self.tb_dir, self.info_dir, self.output_dir, self.extra_data_dir])

        self.device = torch.device("cuda:%d" % self.cuda_id if torch.cuda.is_available() else "cpu")

        if save:
            self.config_name = args.config
            cfg_file = "%s.py" % self.config_name
            shutil.copy(pjoin(BASEPATH, cfg_file), os.path.join(self.info_dir, cfg_file))

