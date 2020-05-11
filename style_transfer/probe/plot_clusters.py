import torch
import os
import sys
import argparse
import importlib
import numpy as np
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH))
sys.path.insert(0, pjoin(BASEPATH, '..'))

from data_loader import get_dataloader

from latent_plot_utils import get_all_plots, get_demo_plots
from trainer import Trainer
from py_utils import to_float, ensure_dirs


def get_all_codes(cfg, output_path):

    print(output_path)
    if os.path.exists(output_path):
        return np.load(output_path, allow_pickle=True)['data'].item()
    ensure_dirs(os.path.dirname(output_path))

    print("start over")
    # Dataloader
    train_loader = get_dataloader(cfg, 'train', shuffle=False)
    test_loader = get_dataloader(cfg, 'test', shuffle=False)

    # Trainer
    trainer = Trainer(cfg)
    trainer.to(cfg.device)
    trainer.resume()

    with torch.no_grad():
        vis_dicts = {}
        for phase, loader in [['train', train_loader],
                              ['test', test_loader]]:

            vis_dict = None
            for t, data in enumerate(loader):
                vis_codes = trainer.get_latent_codes(data)
                if vis_dict is None:
                    vis_dict = {}
                    for key, value in vis_codes.items():
                        vis_dict[key] = [value]
                else:
                    for key, value in vis_codes.items():
                        vis_dict[key].append(value)
            for key, value in vis_dict.items():
                if phase == "test" and key == "content_code":
                    continue
                if key == "meta":
                    secondary_keys = value[0].keys()
                    num = len(value)
                    vis_dict[key] = {
                        secondary_key: [to_float(item) for i in range(num) for item in value[i][secondary_key]]
                        for secondary_key in secondary_keys}
                else:
                    vis_dict[key] = torch.cat(vis_dict[key], 0)
                    vis_dict[key] = vis_dict[key].cpu().numpy()
                    vis_dict[key] = to_float(vis_dict[key].reshape(vis_dict[key].shape[0], -1))
            vis_dicts[phase] = vis_dict

        np.savez_compressed(output_path, data=vis_dicts)
        return vis_dicts


def plot_all(cfg):
    output_path = pjoin(cfg.main_dir, 'test_probe')
    vis_dicts = get_all_codes(cfg, pjoin(output_path, 'output_codes.npz'))
    get_all_plots(vis_dicts, output_path, {}, 0, summary=False,
                  style_cluster_protocols=('tsne'),
                  separate_compute=True)


def plot_demo(cfg):
    BASEPATH = pjoin(os.path.dirname(__file__), '..')
    output_path = pjoin(BASEPATH, "demo_results", "figures")
    vis_dicts = get_all_codes(cfg, pjoin(output_path, 'output_codes.npz'))
    get_demo_plots(vis_dicts, output_path)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--config', type=str, default='config')
    return parser.parse_args()


def main(args):
    config_module = importlib.import_module(args.config)
    config = config_module.Config()
    config.initialize(args)
    plot_demo(config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
