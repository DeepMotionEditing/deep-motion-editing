import models.skeleton as skeleton
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.skeleton import SkeletonConv, SkeletonUnpool, SkeletonPool
from models.Kinematics import ForwardKinematics


class Discriminator(nn.Module):
    def __init__(self, args, topology):
        super(Discriminator, self).__init__()
        self.topologies = [topology]
        self.channel_base = [3]
        self.channel_list = []
        self.joint_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(args.num_layers):
            seq = []
            neighbor_list = skeleton.find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.joint_num[i]
            out_channels = self.channel_base[i+1] * self.joint_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            if i < args.num_layers - 1: bias = False
            else: bias = True

            if i == args.num_layers - 1:
                kernel_size = 16
                padding = 0

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.joint_num[i], kernel_size=kernel_size, stride=2, padding=padding,
                                    padding_mode='reflection', bias=bias))
            if i < args.num_layers - 1: seq.append(nn.BatchNorm1d(out_channels))
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list))
            seq.append(pool)
            if not self.args.patch_gan or i < args.num_layers - 1:
                seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.joint_num.append(len(pool.new_edges) + 1)
            if i == args.num_layers - 1:
                self.last_channel = self.joint_num[-1] * self.channel_base[i+1]

        if not args.patch_gan: self.compress = nn.Linear(in_features=self.last_channel, out_features=1)

    def forward(self, input):
        input = input.reshape(input.shape[0], input.shape[1], -1)
        input = input.permute(0, 2, 1)
        for layer in self.layers:
            input = layer(input)
        if not self.args.patch_gan:
            input = input.reshape(input.shape[0], -1)
            input = self.compress(input)
        # shape = (64, 72, 9)
        return torch.sigmoid(input).squeeze()


class Generator(nn.Module):
    def __init__(self, args, d: Discriminator):
        super(Generator, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.begin = nn.Parameter(data=torch.randn((self.latent_dimension, d.last_channel, args.window_size // (2 ** (args.num_layers - 1)))) )
        self.layers = nn.ModuleList()
        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        for i in range(args.num_layers):
            seq = []
            in_channels = d.channel_list[args.num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = skeleton.find_neighbor(d.topologies[args.num_layers - i - 1], args.skeleton_dist)

            if i != 0 and i != args.num_layers - 1: bias = False
            else: bias = True

            if i != 0: seq.append(nn.Upsample(scale_factor=2, mode=args.upsampling))
            seq.append(SkeletonUnpool(d.pooling_maps[args.num_layers - i - 1]))
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels, joint_num=d.joint_num[args.num_layers - i - 1], kernel_size=kernel_size, stride=1, padding=padding, padding_mode=args.padding_mode, bias=bias))
            if i != 0 and i != args.num_layers - 1: seq.append(nn.BatchNorm1d(out_channels))
            if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input):
        input = input * self.begin
        input = torch.mean(input, dim=1)
        for layer in self.layers:
            input = layer(input)
        return input
