# Classes in this file are mainly borrowed from Jun-Yan Zhu's cycleGAN repository

from torch import optim
from torch import nn
import torch
import random
from torch.optim import lr_scheduler


class GAN_loss(nn.Module):
    def __init__(self, gan_mode, real_lable=1.0, fake_lable=0.0):
        super(GAN_loss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_lable))
        self.register_buffer('fake_label', torch.tensor(fake_lable))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'none':
            self.loss = None
        else:
            raise Exception('Unknown GAN mode')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class Criterion_EE:
    def __init__(self, args, base_criterion, norm_eps=0.008):
        self.args = args
        self.base_criterion = base_criterion
        self.norm_eps = norm_eps

    def __call__(self, pred, gt):
        reg_ee_loss = self.base_criterion(pred, gt)
        if self.args.ee_velo:
            gt_norm = torch.norm(gt, dim=-1)
            contact_idx = gt_norm < self.norm_eps
            extra_ee_loss = self.base_criterion(pred[contact_idx], gt[contact_idx])
        else:
            extra_ee_loss = 0
        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return []

class Criterion_EE_2:
    def __init__(self, args, base_criterion, norm_eps=0.008):
        print('Using adaptive EE')
        self.args = args
        self.base_criterion = base_criterion
        self.norm_eps = norm_eps
        self.ada_para = nn.Linear(15, 15).to(torch.device(args.cuda_device))

    def __call__(self, pred, gt):
        pred = pred.reshape(pred.shape[:-2] + (-1,))
        gt = gt.reshape(gt.shape[:-2] + (-1,))
        pred = self.ada_para(pred)
        reg_ee_loss = self.base_criterion(pred, gt)
        extra_ee_loss = 0
        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return list(self.ada_para.parameters())

class Eval_Criterion:
    def __init__(self, parent):
        self.pa = parent
        self.base_criterion = nn.MSELoss()
        pass

    def __call__(self, pred, gt):
        for i in range(1, len(self.pa)):
            pred[..., i, :] += pred[..., self.pa[i], :]
            gt[..., i, :] += pred[..., self.pa[i], :]
        return self.base_criterion(pred, gt)


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_ee(pos, pa, ees, velo=False, from_root=False):
    pos = pos.clone()
    for i, fa in enumerate(pa):
        if i == 0: continue
        if not from_root and fa == 0: continue
        pos[:, :, i, :] += pos[:, :, fa, :]

    pos = pos[:, :, ees, :]
    if velo:
        pos = pos[:, 1:, ...] - pos[:, :-1, ...]
        pos = pos * 10
    return pos
