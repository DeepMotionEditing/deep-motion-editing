import os
from os.path import join as pjoin
from get_error import full_batch
import numpy as np
from option_parser import try_mkdir
from eval import eval
import argparse


def batch_copy(source_path, suffix, dest_path, dest_suffix=None):
    try_mkdir(dest_path)
    files = [f for f in os.listdir(source_path) if f.endswith('_{}.bvh'.format(suffix))]

    length = len('_{}.bvh'.format(suffix))
    for f in files:
        if dest_suffix is not None:
            cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source_path, f), os.path.join(dest_path, f[:-length] + '_{}.bvh'.format(dest_suffix)))
        else:
            cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source_path, f), os.path.join(dest_path, f[:-length] + '.bvh'))
        os.system(cmd)


if __name__ == '__main__':
    test_characters = ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./pretrained/')
    args = parser.parse_args()
    prefix = args.save_dir

    cross_dest_path = pjoin(prefix, 'results/cross_structure/')
    intra_dest_path = pjoin(prefix, 'results/intra_structure/')
    source_path = pjoin(prefix, 'results/bvh/')

    cross_error = []
    intra_error = []
    for i in range(4):
        print('Batch [{}/4]'.format(i + 1))
        eval(i, prefix)

        print('Collecting test error...')
        if i == 0:
            cross_error += full_batch(0, prefix)
            for char in test_characters:
                batch_copy(os.path.join(source_path, char), 0, os.path.join(cross_dest_path, char))
                batch_copy(os.path.join(source_path, char), 'gt', os.path.join(cross_dest_path, char), 'gt')

        intra_dest = os.path.join(intra_dest_path, 'from_{}'.format(test_characters[i]))
        for char in test_characters:
            for char in test_characters:
                batch_copy(os.path.join(source_path, char), 1, os.path.join(intra_dest, char))
                batch_copy(os.path.join(source_path, char), 'gt', os.path.join(intra_dest, char), 'gt')

        intra_error += full_batch(1, prefix)

    cross_error = np.array(cross_error)
    intra_error = np.array(intra_error)

    cross_error_mean = cross_error.mean()
    intra_error_mean = intra_error.mean()

    os.system('rm -r %s' % pjoin(prefix, 'results/bvh'))

    print('Intra-retargeting error:', intra_error_mean)
    print('Cross-retargeting error:', cross_error_mean)
    print('Evaluation finished!')
