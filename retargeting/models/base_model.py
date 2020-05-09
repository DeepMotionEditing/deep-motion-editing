import os
import torch
import torch.optim
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, args):
        self.args = args
        self.is_train = args.is_train
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.model_save_dir = os.path.join(args.save_dir, 'models')  # save all the checkpoints to save_dir

        if self.is_train:
            from loss_record import LossRecorder
            from torch.utils.tensorboard import SummaryWriter
            self.log_path = os.path.join(args.save_dir, 'logs')
            self.writer = SummaryWriter(self.log_path)
            self.loss_recoder = LossRecorder(self.writer)

        self.epoch_cnt = 0
        self.schedulers = []
        self.optimizers = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def compute_test_result(self):
        """
        After forward, do something like output bvh, get error value
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass


    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def get_scheduler(self, optimizer):
        if self.args.scheduler == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - self.args.n_epochs_origin) / float(self.args.n_epochs_decay + 1)
                return lr_l
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        if self.args.scheduler == 'Step_LR':
            print('Step_LR scheduler set')
            return torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
        if self.args.scheduler == 'Plateau':
            print('Plateau_LR shceduler set')
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5, verbose=True)
        if self.args.scheduler == 'MultiStep':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])

    def setup(self):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.is_train:
            self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

    def epoch(self):
        self.loss_recoder.epoch()
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()
        self.epoch_cnt += 1

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_test_result()
