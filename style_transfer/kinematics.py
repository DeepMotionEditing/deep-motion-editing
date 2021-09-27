import torch
import numpy as np
import math
import copy
import os
import sys
BASEPATH = os.path.dirname(__file__)
from os.path import join as pjoin
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from utils.load_skeleton import Skel


class ForwardKinematics:
    def __init__(self, skel=None):
        if skel is None:
            skel = Skel()
        offset = skel.offset
        topology = skel.topology
        if isinstance(offset, np.ndarray): self.offset = torch.tensor(offset, dtype=torch.float)
        elif isinstance(offset, torch.Tensor): self.offset = offset
        else: raise Exception('Unknown type for offset')
        self.topology = copy.copy(topology)
        self.chosen_joints = skel.chosen_joints

    # rel [B, T, J, 3] relative joint positions --> save needed joints, remove root & flip, [B, (J - 1) * 3, T]
    def trim(self, rel):
        rel = rel[..., self.chosen_joints, :]
        result = rel[..., 1:, :]
        result = result.reshape(result.shape[:2] + (-1,))  # [B, T, (J - 1) * 3]
        result = result.permute(0, 2, 1)  # [B, (J - 1) * 3, T]
        return result

    def forwardX(self, rotations):
        local_pos = self.forward_from_raw(rotations, world=True, quater=True)  # [B, T, J, 3]
        local_pos = self.trim(local_pos)
        return local_pos

    # rotation: [B, J * 4, T]
    def forward_from_raw(self, rotation, world=True, quater=True):
        if quater:
            rotation = rotation.reshape(rotation.shape[0], -1, 4, rotation.shape[-1])
            norm = torch.norm(rotation, dim=2, keepdim=True)
            ones = torch.ones(norm.shape, device=rotation.device) * 1e-10
            norm = torch.where(norm < ones, ones, norm)
            rotation = rotation / norm
        else:
            rotation = rotation.reshape(rotation.shape[0], -1, 3, rotation.shape[-1])
        position = torch.zeros(rotation.shape[:1] + (3, ) + rotation.shape[-1:], device=rotation.device) # should be [B, 3, T]
        return self.forward(rotation, position, world=world, quater=quater)

    '''
    rotation should have shape batch_size * Joint_num * (3/4) * Time
    position should have shape batch_size * 3 * Time
    output have shape batch_size * Time * Joint_num
    '''
    def forward(self, rotation: torch.Tensor, position: torch.Tensor, order='xyz', quater=True, world=True):
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        result = torch.empty(rotation.shape[:-1] + (3, ), device=position.device)

        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            rotation = rotation / 180 * math.pi
            transform = torch.matmul(ForwardKinematics.transform_from_euler(rotation[..., 1], order[1]),
                                     ForwardKinematics.transform_from_euler(rotation[..., 2], order[2]))
            transform = torch.matmul(ForwardKinematics.transform_from_euler(rotation[..., 0], order[0]), transform)

        if self.offset.device != rotation.device:
            self.offset = self.offset.to(rotation.device)

        result[..., 0, :] = position
        for i, pi in enumerate(self.topology):
            if pi == -1:
                assert i == 0
                continue

            result[..., i, :] = torch.matmul(transform[..., pi, :, :], self.offset[i])
            if world: result[..., i, :] += result[..., pi, :]
            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
        return result

    @staticmethod
    def transform_from_euler(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m
    """
    [B, T, J, 3] global positions => [B, J * 3, T], relative pos + glb root
    """
    def global2local(self, x):
        pos = x[:, :, self.chosen_joints]
        rpos = pos[:, :, 0:1] * 1.0 # [B, T, 1, 3]
        pos -= rpos # minus root, relative positions
        ret = torch.cat((pos[:, :, 1:], rpos), dim=-2)
        ret = ret.reshape(ret.shape[:2] + (-1, )) # [B, T, -1]
        ret = ret.permute(0, 2, 1) # [B, J * 3, T]
        return ret


