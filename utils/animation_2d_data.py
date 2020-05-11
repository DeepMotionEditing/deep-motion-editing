import sys
import os
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
from style_transfer.probe.anim_view import visualize


class AnimationData2D:
    def __init__(self, projection):
        self.projection = projection  # [T, J, 2]
        self.style2d = None

    def get_style2d(self):
        if self.style2d is None:
            root = self.projection[..., :1, :].copy()
            relative = self.projection[..., 1:, :].copy() - root
            style2d = np.concatenate([relative, root], axis=-2)
            style2d = style2d.reshape(style2d.shape[:-2] + (-1,)).swapaxes(-1, -2)
            self.style2d = style2d

        return self.style2d

    def get_projection(self):
        return self.projection

    @classmethod
    def from_style2d(cls, style2d):
        style2d = style2d.swapaxes(-1, -2)  # [J * 2, T] -> [T, J * 2]
        style2d = style2d.reshape(style2d.shape[:-1] + (-1, 2))  # [T, J, 2]
        root, relative = style2d[..., -1:, :], style2d[..., :-1, :]
        relative = relative + root
        projection = np.concatenate([root, relative], axis=-2)
        return cls(projection)

    @classmethod
    def from_openpose_json(cls, json_dir, scale=0.07, smooth=True):
        json_files = sorted(os.listdir(json_dir))
        length = len(json_files) // 4 * 4
        json_files = json_files[:length]
        json_files = [os.path.join(json_dir, x) for x in json_files]

        motion = []
        joint_map = {
            0: 8,
            1: 12, 2: 13, 3: 14, 4: 19,
            5: 9, 6: 10, 7: 11, 8: 22,
            # 9 is somewhere between 0 & 10
            10: 1,
            # 11 is somewhere between 10 and 12
            12: 0,
            13: 5, 14: 6, 15: 7,  # 16 is a little bit further
            17: 2, 18: 3, 19: 4,  # 20 is a little bit further
        }

        num_joints = 21
        start = False

        for path in json_files:
            with open(path) as f:
                joint_dict = json.load(f)
                if len(joint_dict['people']) == 0:
                    if start:
                        raw_joint = motion[-1]
                        motion.append(raw_joint)
                    else:
                        continue
                start = True
                body_joint = np.array(joint_dict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:, :2]
                lhand_joint = np.array(joint_dict['people'][0]['hand_left_keypoints_2d']).reshape((-1, 3))[:, :2]
                rhand_joint = np.array(joint_dict['people'][0]['hand_right_keypoints_2d']).reshape((-1, 3))[:, :2]
                raw_joint = np.concatenate([body_joint, lhand_joint, rhand_joint], axis=-2)
                if len(motion) > 0:
                    raw_joint[np.where(raw_joint == 0)] = motion[-1][np.where(raw_joint == 0)]
                motion.append(raw_joint)

        for i in range(len(motion) - 1, 0, -1):
            motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]

        motion = np.stack(motion, axis=0)
        # motion: [T, J, 2]

        trans_motion = np.zeros((motion.shape[0], num_joints, 2))
        for i in range(num_joints):
            if i in [9, 11, 12, 16, 20]:
                continue
            else:
                trans_motion[:, i, :] = motion[:, joint_map[i], :]

        trans_motion[:, 12, :] = (motion[:, 15, :] + motion[:, 16, :]) / 2.0
        trans_motion[:, 16, :] = motion[:, 35, :]  # 25 + 10
        trans_motion[:, 20, :] = motion[:, 56, :]  # 25 + 21 + 10

        trans_motion[:, 9, :] = (trans_motion[:, 0, :] + trans_motion[:, 10, :]) / 2
        trans_motion[:, 11, :] = (trans_motion[:, 10, :] + trans_motion[:, 12, :]) / 2

        motion = trans_motion
        motion[:, :, 1] = -motion[:, :, 1]  # upside-down
        motion[:, :, :] -= motion[0:1, 0:1, :]  # start from zero

        if smooth:
            motion = gaussian_filter1d(motion, sigma=2, axis=0)

        motion = motion * scale
        return cls(motion)


def test():
    style2d = np.random.rand(42, 60)
    anim = AnimationData2D.from_style2d(style2d)
    bla = anim.get_style2d()

    print(np.sum(style2d - bla))

    bla = {}
    for num in [27, 32, 95]:
        anim2d = AnimationData2D.from_openpose_json(f'../../data/treadmill/json_inputs/{num}')
        bla[str(num)] = {"motion": anim2d.get_projection(), "foot_contact": None}

    visualize(bla)


if __name__ == '__main__':
    test()
