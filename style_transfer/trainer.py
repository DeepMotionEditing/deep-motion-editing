import os
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler

from model import Model
from py_utils import update_dict


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None or len(gen_models) == 0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, config, it=-1):
    lr_policy = config.lr_policy
    if lr_policy is None or lr_policy == 'constant':
        scheduler = None  # constant scheduler
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size,
                                        gamma=config.step_gamma, last_epoch=it)
    else:
        return NotImplementedError('%s not implemented', lr_policy)
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


class Trainer(nn.Module):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.model = Model(cfg)
        lr_gen = cfg.lr_gen
        lr_dis = cfg.lr_dis
        dis_params = list(self.model.dis.parameters())
        gen_params = list(self.model.gen.parameters())
        self.dis_opt = torch.optim.RMSprop(
            [p for p in dis_params if p.requires_grad],
            lr=lr_gen, weight_decay=cfg.weight_decay, eps=1e-4)
        self.gen_opt = torch.optim.RMSprop(
            [p for p in gen_params if p.requires_grad],
            lr=lr_dis, weight_decay=cfg.weight_decay, eps=1e-4)
        self.dis_scheduler = get_scheduler(self.dis_opt, cfg)
        self.gen_scheduler = get_scheduler(self.gen_opt, cfg)
        self.apply(weights_init(cfg.weight_init))

        self.model_dir = cfg.model_dir
        self.cfg = cfg

        self.loss_dict = {}

    def gen_update(self, co_data, cl_data, multigpus=None):
        self.gen_opt.zero_grad()
        gen_loss_dict = self.model(co_data, cl_data, 'gen_update')
        update_dict(self.loss_dict, gen_loss_dict)
        self.gen_opt.step()
        return self.loss_dict['gen_acc_all'].item()

    def dis_update(self, co_data, cl_data):
        self.dis_opt.zero_grad()
        dis_loss_dict = self.model(co_data, cl_data, 'dis_update')
        update_dict(self.loss_dict, dis_loss_dict)
        self.dis_opt.step()
        return self.loss_dict['dis_acc_all'].item()

    def test(self, co_data, cl_data, status):
        return self.model.test(co_data, cl_data, status)

    def test_rec(self, data):
        return self.model.test_rec(data)

    def get_latent_codes(self, data):
        return self.model.get_latent_codes(data)

    def resume(self):
        model_dir = self.model_dir

        last_model_name = get_model_list(model_dir, "gen")
        if last_model_name is None:
            print('Initialize from 0')
            return 0

        state_dict = torch.load(last_model_name, map_location=self.cfg.device)
        self.model.gen.load_state_dict(state_dict['gen'])
        last_model_name = get_model_list(model_dir, "dis")
        state_dict = torch.load(last_model_name, map_location=self.cfg.device)
        self.model.dis.load_state_dict(state_dict['dis'])

        optim_name = os.path.join(model_dir, 'optimizer.pt')
        state_dict = torch.load(optim_name, map_location=self.cfg.device)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        iterations = int(last_model_name[-11:-3])
        self.dis_scheduler = get_scheduler(self.dis_opt, self.cfg, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, self.cfg, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, iterations):
        gen_name = os.path.join(self.model_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(self.model_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(self.model_dir, 'optimizer.pt')
        torch.save({'gen': self.model.gen.state_dict()}, gen_name)
        torch.save({'dis': self.model.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)

