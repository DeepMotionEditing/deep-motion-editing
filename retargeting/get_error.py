import sys
import os
from option_parser import get_std_bvh
sys.path.append("../utils")
import BVH as BVH
import numpy as np
from datasets.bvh_parser import BVH_file
import Animation


def full_batch(suffix, prefix):
    res = []
    chars = ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']
    for char in chars:
        res.append(batch(char, suffix, prefix))
    return res


def batch(char, suffix, prefix):
    input_path = os.path.join(prefix, 'results/bvh')

    all_err = []
    ref_file = get_std_bvh(dataset=char)
    ref_file = BVH_file(ref_file)
    height = ref_file.get_height()

    test_num = 0

    new_p = os.path.join(input_path, char)

    files = [f for f in os.listdir(new_p) if
             f.endswith('_{}.bvh'.format(suffix)) and not f.endswith('_gt.bvh') and 'fix' not in f and not f.endswith('_input.bvh')]

    for file in files:
        file_full = os.path.join(new_p, file)
        anim, names, _ = BVH.load(file_full)
        test_num += 1
        index = []
        for i, name in enumerate(names):
            if 'virtual' in name:
                continue
            index.append(i)

        file_ref = file_full[:-6] + '_gt.bvh'
        anim_ref, _, _ = BVH.load(file_ref)

        pos = Animation.positions_global(anim)  # [T, J, 3]
        pos_ref = Animation.positions_global(anim_ref)

        pos = pos[:, index, :]
        pos_ref = pos_ref[:, index, :]

        err = (pos - pos_ref) * (pos - pos_ref)
        err /= height ** 2
        err = np.mean(err)
        all_err.append(err)

    all_err = np.array(all_err)
    return all_err.mean()
