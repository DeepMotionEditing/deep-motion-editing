from torch.utils.data import Dataset
import os
import sys
sys.path.append("../utils")
import numpy as np
import torch
from Quaternions import Quaternions
from option_parser import get_std_bvh


class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args):
        super(MotionData, self).__init__()
        name = args.dataset
        file_path = './datasets/Mixamo/{}.npy'.format(name)

        if args.debug:
            file_path = file_path[:-4] + '_debug' + file_path[-4:]

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        self.std_bvh = get_std_bvh(args)
        self.args = args
        self.data = []
        self.motion_length = []
        motions = np.load(file_path, allow_pickle=True)
        motions = list(motions)
        new_windows = self.get_windows(motions)
        self.data.append(new_windows)
        self.data = torch.cat(self.data)
        self.data = self.data.permute(0, 2, 1)

        if args.normalization == 1:
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.var = torch.var(self.data, (0, 2), keepdim=True)
            self.var = self.var ** (1/2)
            idx = self.var < 1e-5
            self.var[idx] = 1
            self.data = (self.data - self.mean) / self.var
        else:
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.mean.zero_()
            self.var = torch.ones_like(self.mean)

        train_len = self.data.shape[0] * 95 // 100
        self.test_set = self.data[train_len:, ...]
        self.data = self.data[:train_len, ...]
        self.data_reverse = torch.tensor(self.data.numpy()[..., ::-1].copy())

        self.reset_length_flag = 0
        self.virtual_length = 0
        print('Window count: {}, total frame (without downsampling): {}'.format(len(self), self.total_frame))

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]
        if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
            return self.data[item]
        else:
            return self.data_reverse[item]

    def get_windows(self, motions):
        new_windows = []

        for motion in motions:
            self.total_frame += motion.shape[0]
            motion = self.subsample(motion)
            self.motion_length.append(motion.shape[0])
            step_size = self.args.window_size // 2
            window_size = step_size * 2
            n_window = motion.shape[0] // step_size - 1
            for i in range(n_window):
                begin = i * step_size
                end = begin + window_size

                new = motion[begin:end, :]
                if self.args.rotation == 'quaternion':
                    new = new.reshape(new.shape[0], -1, 3)
                    rotations = new[:, :-1, :]
                    rotations = Quaternions.from_euler(np.radians(rotations)).qs
                    rotations = rotations.reshape(rotations.shape[0], -1)
                    positions = new[:, -1, :]
                    positions = np.concatenate((new, np.zeros((new.shape[0], new.shape[1], 1))), axis=2)
                    new = np.concatenate((rotations, new[:, -1, :].reshape(new.shape[0], -1)), axis=1)

                new = new[np.newaxis, ...]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)

    def subsample(self, motion):
        return motion[::2, :]

    def denormalize(self, motion):
        if self.args.normalization:
            if self.var.device != motion.device:
                self.var = self.var.to(motion.device)
                self.mean = self.mean.to(motion.device)
            ans = motion * self.var + self.mean
        else: ans = motion
        return ans
