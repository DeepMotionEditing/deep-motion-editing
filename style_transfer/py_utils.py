import os
import numpy as np
import torch

def merge_dict(dict_list):
    ret = {}
    for dict in dict_list:
        for key, value in dict.items():
            try:
                ret[key]
            except KeyError:
                ret[key] = 0.0
            ret[key] += value
    return ret


def update_dict(old_dict, new_dict):
    for key, value in new_dict.items():
        old_dict[key] = value


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        print("Create folder ", path)
        os.makedirs(path)
    else:
        print(path, " already exists.")


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def write_loss(iterations, trainer, train_writer):
    for key, value in trainer.loss_dict.items():
        train_writer.add_scalar(key, value, iterations + 1)


def print_composite(data, beg=""):
    if isinstance(data, dict):
        print(f'{beg} dict, size = {len(data)}')
        for key, value in data.items():
            print(f'  {beg}{key}:')
            print_composite(value, beg + "    ")
    elif isinstance(data, list):
        print(f'{beg} list, len = {len(data)}')
        for i, item in enumerate(data):
            print(f'  {beg}item {i}')
            print_composite(item, beg + "    ")
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        print(f'{beg} array of size {data.shape}')
    else:
        print(f'{beg} {data}')


def to_float(item):
    if isinstance(item, torch.Tensor):
        item = item.to('cpu').numpy()
    if isinstance(item, np.ndarray):
        if len(item.reshape(-1)) == 1:
            item = float(item)
    return item

if __name__ == "__main__":
    bla = np.random.rand(1, 1, 1)
    bla = torch.tensor(bla)
    cla = np.random.rand(2, 3)
    cla = torch.tensor(cla)
    print(to_float(bla))
    print(to_float(cla))
    print(to_float("bla"))
