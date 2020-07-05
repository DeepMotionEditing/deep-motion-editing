import os
import numpy as np
import copy
from datasets.bvh_parser import BVH_file
from datasets.motion_dataset import MotionData
from option_parser import get_args, try_mkdir


def collect_bvh(data_path, character, files):
    print('begin {}'.format(character))
    motions = []

    for i, motion in enumerate(files):
        if not os.path.exists(data_path + character + '/' + motion):
            continue
        file = BVH_file(data_path + character + '/' + motion)
        new_motion = file.to_tensor().permute((1, 0)).numpy()
        motions.append(new_motion)

    save_file = data_path + character + '.npy'

    np.save(save_file, motions)
    print('Npy file saved at {}'.format(save_file))


def write_statistics(character, path):
    args = get_args()
    new_args = copy.copy(args)
    new_args.data_augment = 0
    new_args.dataset = character

    dataset = MotionData(new_args)

    mean = dataset.mean
    var = dataset.var
    mean = mean.cpu().numpy()[0, ...]
    var = var.cpu().numpy()[0, ...]

    np.save(path + '{}_mean.npy'.format(character), mean)
    np.save(path + '{}_var.npy'.format(character), var)


def copy_std_bvh(data_path, character, files):
    """
    copy an arbitrary bvh file as a static information (skeleton's offset) reference
    """
    cmd = 'cp \"{}\" ./datasets/Mixamo/std_bvhs/{}.bvh'.format(data_path + character + '/' + files[0], character)
    os.system(cmd)


if __name__ == '__main__':
    prefix = './datasets/Mixamo/'
    characters = [f for f in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, f))]
    if 'std_bvhs' in characters: characters.remove('std_bvhs')
    if 'mean_var' in characters: characters.remove('mean_var')

    try_mkdir(os.path.join(prefix, 'std_bvhs'))
    try_mkdir(os.path.join(prefix, 'mean_var'))

    for character in characters:
        data_path = os.path.join(prefix, character)
        files = sorted([f for f in os.listdir(data_path) if f.endswith(".bvh")])

        collect_bvh(prefix, character, files)
        copy_std_bvh(prefix, character, files)
        write_statistics(character, './datasets/Mixamo/mean_var/')
