import sys
import os
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

import argparse
import numpy as np
import scipy.ndimage.filters as filters
from load_skeleton import Skel
from Quaternions_old import Quaternions
from Pivots import Pivots
import BVH
from probe.anim_view import visualize


def forward_rotations(skel, rotations, rtpos=None, trim=True):
    """
    input: rotations [T, J, 4], rtpos [T, 3]
    output: positions [T, J, 3]
    """
    transforms = Quaternions(rotations).transforms()  # [..., J, 3, 3]
    glb = np.zeros(rotations.shape[:-1] + (3,))  # [T, J, 3]
    if rtpos is not None:
        glb[..., 0, :] = rtpos
    for i, pi in enumerate(skel.topology):
        if pi == -1:
            continue
        glb[..., i, :] = np.matmul(transforms[..., pi, :, :],
                                   skel.offset[i])
        glb[..., i, :] += glb[..., pi, :]
        transforms[..., i, :, :] = np.matmul(transforms[..., pi, :, :],
                                             transforms[..., i, :, :])
    if trim:
        glb = glb[..., skel.chosen_joints, :]
    return glb


def rotate_coordinates(local3d, angles):
    """
    Rotate xyz coordinates from given view_angles.
    local3d: numpy array. Unit LOCAL xyz vectors
    angles: tuple of length 3. Rotation angles around each GLOBAL axis.
    """
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    mat33_x = np.array([
        [1, 0, 0],
        [0, cx, sx],
        [0, -sx, cx]
    ], dtype='float')

    mat33_y = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype='float')

    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x @ mat33_y @ mat33_z
    return local3d


def get_local3d(local_x, view_angle=None):
    """
    Get the unit vectors for local rectangular coordinates for given 3D motion
    :param local_x: local x axis, (B *) [*, 0, *]
    :return: numpy array. unit vectors for local rectangular coordinates's , shape (3, 3).
    """
    local_y = np.zeros_like(local_x)  # [(B,) 3]
    local_y[..., :] = np.array([0, 1, 0])
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z, axis=-1, keepdims=True)

    local = np.stack([local_x, local_y, local_z], axis=-2)

    if view_angle is not None:
        local = rotate_coordinates(local, view_angle)

    return local


def motion_projection(motion, local_x, view_angle=None):
    """
    motion: motion in relative joint positions & global root positions
                   [(B,) T, (J - 1) + 1, 3]
    local_x: [(B,) 3], local x-axis
    view_angle: [3], the angles to rotate
    output: motion_proj [(B,) J * 2, T]
    """

    local = get_local3d(local_x, view_angle) # [(B,) 3, 3]

    T = motion.shape[-1]
    # proj on xy-plane
    # motion_proj = (local[[0, 1], :] @ motion)  this used to be [2, 3] @ [J, 3, T]
    # but T doesn't matter here ... what we care is the "3", using [T, J, 3, 1] would also be OK
    motion = motion[..., np.newaxis] # [(B,) T, J, 3, 1]
    motion_proj = local[..., np.newaxis, np.newaxis, [0, 1], :] @ motion  # [(B,), 1, 1, 2, 3] @ [(B,), T, J, 3, 1] => [(B,), T, J, 2, 1]
    motion_proj = motion_proj.reshape(motion_proj.shape[:-3] + (-1, ))  # [(B,) T, -1]
    motion_proj = motion_proj.swapaxes(-1, -2)  # [(B,) J * 2, T]

    return motion_proj


def foot_contact_from_positions(positions, fid_l=(3, 4), fid_r=(7, 8)):
    """
    positions: [T, J, 3], trimmed (only "chosen_joints")
    fid_l, fid_r: indices of feet joints (in "chosen_joints")
    """
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    velfactor = np.array([0.05, 0.05])
    feet_contact = []
    for fid_index in [fid_l, fid_r]:
        foot_vel = (positions[1:, fid_index] - positions[:-1, fid_index]) ** 2  # [T - 1, 2, 3]
        foot_vel = np.sum(foot_vel, axis=-1)  # [T - 1, 2]
        foot_contact = (foot_vel < velfactor).astype(np.float)
        feet_contact.append(foot_contact)
    feet_contact = np.concatenate(feet_contact, axis=-1)  # [T - 1, 4]
    feet_contact = np.concatenate((feet_contact[0:1].copy(), feet_contact), axis=0)

    return feet_contact  # [T, 4]


def phase_from_ft(foot_contact, is_debug=False):
    """
    foot_contact: [T, 4] -> take joints 0, 2 as standards
    phase = left foot in contact (0) --> right foot in contact (pi) --> left foot in contact (2pi),
            in range [0, 2pi)
    """
    num_circles = 0
    circle_length = 0
    total_length = len(foot_contact)
    ft = foot_contact[:, [0, 2]].astype(np.int)
    ft_start = np.zeros((total_length, 2))
    phases = np.zeros((total_length, 1))

    """
    calculate the average "half-phase length"
    find the first and last "01" pattern
    """
    for j in range(2):
        for i in range(1, total_length):
            ft_start[i, j] = (ft[i - 1, j] == 0 and ft[i, j] == 1)
    if is_debug:
        print('ft_start,', ft_start)

    last, beg_i = -1, -1
    starts = []
    for i in range(total_length):
        if ft_start[i, 0] or ft_start[i, 1]:
            if last != -1:
                num_circles += 1
                circle_length += i - last
            else:
                beg_i = i
            last = i
            starts.append(i)

    avg_circle = 0 if num_circles == 0 else circle_length * 1.0 / num_circles
    if is_debug:
        print("%d circles, total length = %d, avg length = %.3lf" % (num_circles, circle_length, avg_circle))

    if len(starts) == 0:  # phase never changed
        return phases

    """[0, beg_i - 1]: first incomplete circle"""
    prev_pos = min(0, beg_i - avg_circle)
    prev_val = 0 if ft_start[beg_i, 1] == 1 else 1  # 0 if next step is on the right
    cir_i = 0
    next_pos = starts[cir_i]

    for i in range(total_length):
        if i == next_pos:
            prev_pos = next_pos
            prev_val = 1 - prev_val
            cir_i += 1
            if cir_i >= len(starts):
                next_pos = max(total_length + 1, next_pos + avg_circle)
            else:
                next_pos = starts[cir_i]
        phases[i] = prev_val + (i - prev_pos) * 1.0 / (next_pos - prev_pos)

    phases *= np.pi
    if is_debug:
        print('phases:', phases)
    return phases


def across_from_glb(positions, hips=(2, 6), sdrs=(14, 18)):
    """
    positions: positions [T, J, 3], trimmed (only "chosen_joints")
    hips, sdrs: left/right hip joints, left/right shoulder joints
    output: local x-axis for each frame [T, 3]
    """
    across = positions[..., hips[0], :] - positions[..., hips[1], :] + \
             positions[..., sdrs[0], :] - positions[..., sdrs[1], :]  # [T, 3]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    return across


def y_rotation_from_positions(positions, hips=(2, 6), sdrs=(14, 18)):
    """
    input: positions [T, J, 3]
    output: quaters: [T, 1, 4], quaternions that rotate the character around the y-axis to face [0, 0, 1]
            pivots: [T, 1] in [0, 2pi], the angle from [0, 0, 1] to the current facing direction
    """
    across = across_from_glb(positions, hips=hips, sdrs=sdrs)
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.tile(np.array([0, 0, 1]), forward.shape[:-1] + (1, ))
    quaters = Quaternions.between(forward, target)[..., np.newaxis, :]  # [T, 4] -> [T, 1, 4]
    pivots = Pivots.from_quaternions(-quaters).ps  # from "target"[0, 0, 1] to current facing direction "forward"
    return quaters, pivots


class AnimationData:
    """
    Canonical Representation:
        Skeleton
        [T, Jo * 4 + 4 global params + 4 foot_contact]
    """
    def __init__(self, full, skel=None, frametime=1/30):
        if skel is None:
            skel = Skel()
        self.skel = skel
        self.frametime = frametime
        self.len = len(full)
        self.rotations = full[:, :-8].reshape(self.len, -1, 4)  # [T, Jo, 4]
        assert self.rotations.shape[1] == len(self.skel.topology), "Rotations do not match the skeleton."
        self.rotations /= np.sqrt(np.sum(self.rotations ** 2, axis=-1))[..., np.newaxis]
        self.rt_pos = full[:, -8:-5]  # [T, 3]
        self.rt_rot = full[:, -5:-4]  # [T, 1]
        self.foot_contact = full[:, -4:]  # [T, 4]
        self.full = np.concatenate([self.rotations.reshape(self.len, -1), self.rt_pos, self.rt_rot, self.foot_contact], axis=-1)
        self.phases = None  # [T, 1]
        self.local_x = None  # [3]
        self.positions_for_proj = None  # [T, (J - 1) + 1, 3], trimmed and not forward facing
        self.global_positions = None


    def get_full(self):
        return self.full

    def get_root_positions(self):
        return self.rt_pos

    def get_original_rotations(self, rt_rot=None):
        if rt_rot is None:
            rt_rot = self.rt_rot
        yaxis_rotations = Quaternions(np.array(Pivots(rt_rot).quaternions()))
        rt_rotations = Quaternions(self.rotations[:, :1])  # [T, 1, 4]
        rt_rotations = np.array(yaxis_rotations * rt_rotations)
        rt_rotations /= np.sqrt((rt_rotations ** 2).sum(axis=-1))[..., np.newaxis]
        return np.concatenate((rt_rotations, self.rotations[:, 1:]), axis=1)  # [T, J, 4]

    def get_foot_contact(self, transpose=False):
        if transpose:
            return self.foot_contact.transpose(1, 0)  # [4, T]
        else:
            return self.foot_contact

    def get_phases(self):
        if self.phases is None:
            self.phases = phase_from_ft(self.foot_contact)
        return self.phases

    def get_local_x(self):
        if self.local_x is None:
            forward_pivot = np.mean(self.rt_rot, axis=0)  # [T, 1] -> [1]
            forward_dir = Pivots(forward_pivot).directions()
            self.local_x = np.cross(np.array((0, 1, 0)), forward_dir).reshape(-1)
        return self.local_x

    def get_content_input(self):
        rotations = self.rotations.reshape(self.len, -1)  # [T, Jo * 4]
        return np.concatenate((rotations, self.rt_pos, self.rt_rot), axis=-1).transpose(1, 0)  # [Jo * 4 + 3 + 1, T]

    def get_style3d_input(self):
        pos3d = forward_rotations(self.skel, self.rotations, trim=True)[:, 1:]  # [T, J - 1, 3]
        pos3d = pos3d.reshape((len(pos3d), -1))  # [T, (J - 1) * 3]
        return np.concatenate((pos3d, self.rt_pos, self.rt_rot), axis=-1).transpose(1, 0)  # [(J - 1) * 3 + 3 + 1, T]

    def get_projections(self, view_angles, scales=None):
        if self.positions_for_proj is None:
            rotations = self.get_original_rotations()
            positions = forward_rotations(self.skel, rotations, trim=True)[:, 1:]  # [T, J - 1, 3]
            positions = np.concatenate((positions, self.rt_pos[:, np.newaxis, :]), axis=1)  # [T, J, 3]
            self.positions_for_proj = positions.copy()
        else:
            positions = self.positions_for_proj.copy()
        projections = []
        if scales is None:
            scales = np.ones((len(view_angles)))
        for angle, scale in zip(view_angles, scales):
            projections.append(motion_projection(positions, self.get_local_x(), angle) * scale)
        projections = np.stack(projections, axis=-3)  # [V, J * 2, T]
        return projections

    def get_global_positions(self, trim=True):  # for visualization
        if not trim:
            return forward_rotations(self.skel, self.get_original_rotations(), rtpos=self.rt_pos, trim=False)
        if self.global_positions is None:
            rotations = self.get_original_rotations()
            positions = forward_rotations(self.skel, rotations, rtpos=self.rt_pos, trim=True)
            self.global_positions = positions
        return self.global_positions

    def get_velocity_factor(self):
        positions = forward_rotations(self.skel, self.get_original_rotations(), trim=True)[:, 1:]  # [T, J - 1, 3]
        velocity = positions[1:] - positions[:-1]  # [T - 1, J - 1, 3]
        velocity = np.sqrt(np.sum(velocity ** 2, axis=-1))  # [T - 1, J - 1]
        max_velocity = np.max(velocity, axis=-1)  # [T - 1]
        velocity_factor = np.mean(max_velocity)
        return velocity_factor

    def get_BVH(self, forward=True):
        rt_pos = self.rt_pos  # [T, 3]
        rt_rot = self.rt_rot  # [T, 1]
        if forward:  # choose a direction in [z+, x+, z-, x-], which is closest to "forward", as the new z+

            directions = np.array(range(4)) * np.pi * 0.5  # [0, 1, 2, 3] * 0.5pi
            diff = rt_rot[np.newaxis, :] - directions[:, np.newaxis, np.newaxis]  # [1, T, 1] - [4, 1, 1]
            diff = np.minimum(np.abs(diff), 2.0 * np.pi - np.abs(diff))
            diff = np.sum(diff, axis=(-1, -2))  # [4, T, 1] -> [4]

            new_forward = np.argmin(diff)
            rt_rot -= new_forward * np.pi * 0.5

            for d in range(new_forward):
                tmp = rt_pos[..., 0].copy()
                rt_pos[..., 0] = -rt_pos[..., 2].copy()
                rt_pos[..., 2] = tmp

        rotations = self.get_original_rotations(rt_rot=rt_rot)

        rest, names, _ = self.skel.rest_bvh
        anim = rest.copy()
        anim.positions = anim.positions.repeat(self.len, axis=0)
        anim.positions[:, 0, :] = rt_pos
        anim.rotations.qs = rotations

        return (anim, names, self.frametime)

    @classmethod
    def from_network_output(cls, input):
        input = input.transpose(1, 0)
        input = np.concatenate((input, np.zeros((len(input), 4))), axis=-1)
        return cls(input)

    @classmethod
    def from_rotations_and_root_positions(cls, rotations, root_positions, skel=None, frametime=1/30):
        """
        rotations: [T, J, 4]
        root_positions: [T, 3]
        """
        if skel is None:
            skel = Skel()

        rotations /= np.sqrt(np.sum(rotations ** 2, axis=-1))[..., np.newaxis]
        global_positions = forward_rotations(skel, rotations, root_positions, trim=True)
        foot_contact = foot_contact_from_positions(global_positions, fid_l=skel.fid_l, fid_r=skel.fid_r)
        quaters, pivots = y_rotation_from_positions(global_positions, hips=skel.hips, sdrs=skel.sdrs)

        root_rotations = Quaternions(rotations[:, 0:1, :].copy())  # [T, 1, 4]
        root_rotations = quaters * root_rotations  # facing [0, 0, 1]
        root_rotations = np.array(root_rotations).reshape((-1, 1, 4))  # [T, 1, 4]
        rotations[:, 0:1, :] = root_rotations

        full = np.concatenate([rotations.reshape((len(rotations), -1)), root_positions, pivots, foot_contact], axis=-1)
        return cls(full, skel, frametime)

    @classmethod
    def from_BVH(cls, filename, downsample=4, skel=None, trim_scale=None):
        anim, names, frametime = BVH.load(filename)
        anim = anim[::downsample]
        if trim_scale is not None:
            length = (len(anim) // trim_scale) * trim_scale
            anim = anim[:length]
        rotations = np.array(anim.rotations)  # [T, J, 4]
        root_positions = anim.positions[:, 0, :]
        return cls.from_rotations_and_root_positions(rotations, root_positions, skel=skel, frametime=frametime * downsample)


def parse_args():
    parser = argparse.ArgumentParser("test")
    parser.add_argument('--bvh_in', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)

    return parser.parse_args()


def test_all(args):

    def mse(a, b):
        return np.sum((a - b) ** 2)

    def test_phase_from_ft():
        pace = np.zeros((100, 1), dtype=np.int)
        pace[::8] = 1
        left = pace[:-4]
        right = pace[4:]
        phase_from_ft(np.concatenate([left, left, right, right], axis=-1), is_debug=True)

    def BVH_and_back(filename):
        anim, names, frametime = BVH.load(filename)
        anim = anim[::4]
        rotations = np.array(anim.rotations)  # [T, J, 4]
        root_positions = anim.positions[:, 0, :]

        anim_a = AnimationData.from_BVH(filename)
        rotations = rotations / np.sqrt(np.sum(rotations ** 2, axis=-1))[..., np.newaxis]
        print(f'rotations: {mse(anim_a.get_original_rotations(), rotations)}')
        print(f'root_positions: {mse(anim_a.get_root_positions(), root_positions)}')

        content_input = anim_a.get_content_input()
        style3d_input = anim_a.get_style3d_input()
        view_points = ()
        for i in range(7):
            view_points += ((0, -np.pi / 2 + i * np.pi / 6, 0), )
        view_points = ()
        scales = ()
        for i in range(4):
            view_points += ((0, -np.pi / 2 + float(np.random.rand(1)) * np.pi, 0), )
            scales += (float(np.random.rand(1)) * 0.4 + 0.8, )
        style2d_input = anim_a.get_projections(view_points, scales)

        print(f'content {content_input.shape}, style3d {style3d_input.shape}, style2d {style2d_input.shape}')

        foot_contact = anim_a.get_foot_contact()
        T = content_input.shape[-1]
        inplace_no_rot = style3d_input.transpose(1, 0)[:, :-4].reshape(T, -1, 3)
        inplace_no_rot = np.concatenate((np.zeros((T, 1, 3)), inplace_no_rot), axis=1)
        inplace = anim_a.positions_for_proj[:, :-1, :]
        inplace = np.concatenate((np.zeros((T, 1, 3)), inplace), axis=1)
        original = anim_a.get_global_positions()
        print(f'inplace no rot {inplace_no_rot.shape}, inplace {inplace.shape}, original {original.shape}')

        """
        visualize({
            "inplace_no_rot": {"motion": inplace_no_rot, "foot_contact": foot_contact},
            "inplace": {"motion": inplace, "foot_contact": foot_contact},
            "original": {"motion": original, "foot_contact": foot_contact},
        })
        """

        motion_proj = {}
        for (view_point, scale, proj) in zip(view_points, scales, style2d_input):  # [V, J * 2, T]
            proj = proj.copy().transpose(1, 0).reshape(T, -1, 2)  # [T, J, 2]
            proj = np.concatenate([proj[:, -1:], proj[:, :-1]], axis=1)
            ori_proj = np.concatenate([proj[:, :1], proj[:, 1:] + proj[:, :1].copy()], axis=1)
            proj[:, :1] = 0
            motion_proj[f'angle: {(view_point[1] / np.pi * 180):3f} scale: {scale:3f}'] = {"motion": ori_proj, "foot_contact": foot_contact}
            """
            visualize({
                "inplace_proj": {"motion": proj, "foot_contact": foot_contact},
                "original_proj": {"motion": ori_proj, "foot_contact": foot_contact}
            })
            """
        visualize(motion_proj)

        BVH.save("bla.bvh", *anim_a.get_BVH())

    def check_velocity(dataset):
        skel = Skel()
        motions, labels, metas = dataset["motion"], dataset["style"], dataset["meta"]
        style_names = list(set(metas["style"]))
        content_names = list(set(metas["content"]))
        info = {content: {style: [] for style in style_names} for content in content_names}
        for i, motion in enumerate(motions):
            anim = AnimationData(motion, skel=skel)
            vel = anim.get_velocity_factor()
            info[metas["content"][i]][metas["style"][i]].append(vel)

        for content in info:
            all = []
            for style in info[content]:
                all += info[content][style]
                info[content][style] = np.mean(info[content][style])
            info[content]["all"] = np.mean(all)

        with open("probe_velocity.csv", "w") as f:
            columns = ['all'] + style_names
            f.write(',' + ','.join(columns) + '\n')
            for content in info:
                values = [f'{info[content][key]}' for key in columns]
                f.write(','.join([content] + values) + '\n')

    dataset = np.load(args.dataset, allow_pickle=True)["trainfull"].item()
    check_velocity(dataset)

    # BVH_and_back(args.bvh_in)


if __name__ == '__main__':
    args = parse_args()
    test_all(args)

