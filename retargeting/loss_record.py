from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


class SingleLoss:
    def __init__(self, name: str, writer: SummaryWriter):
        self.name = name
        self.loss_step = []
        self.loss_epoch = []
        self.loss_epoch_tmp = []
        self.writer = writer

    def add_scalar(self, val, step=None):
        if step is None: step = len(self.loss_step)
        self.loss_step.append(val)
        self.loss_epoch_tmp.append(val)
        self.writer.add_scalar('Train/step_' + self.name, val, step)

    def epoch(self, step=None):
        if step is None: step = len(self.loss_epoch)
        loss_avg = sum(self.loss_epoch_tmp) / len(self.loss_epoch_tmp)
        self.loss_epoch_tmp = []
        self.loss_epoch.append(loss_avg)
        self.writer.add_scalar('Train/epoch_' + self.name, loss_avg, step)

    def save(self, path):
        loss_step = np.array(self.loss_step)
        loss_epoch = np.array(self.loss_epoch)
        np.save(path + self.name + '_step.npy', loss_step)
        np.save(path + self.name + '_epoch.npy', loss_epoch)


class LossRecorder:
    def __init__(self, writer: SummaryWriter):
        self.losses = {}
        self.writer = writer

    def add_scalar(self, name, val, step=None):
        if isinstance(val, torch.Tensor): val = val.item()
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer)
        self.losses[name].add_scalar(val, step)

    def epoch(self, step=None):
        for loss in self.losses.values():
            loss.epoch(step)

    def save(self, path):
        for loss in self.losses.values():
            loss.save(path)
