import torch
import torch.nn as nn
import numpy as np
import math


class ForwardKinematics:
    def __init__(self, args, edges):
        self.topology = [-1] * (len(edges) + 1)
        self.rotation_map = []
        for i, edge in enumerate(edges):
            self.topology[edge[1]] = edge[0]
            self.rotation_map.append(edge[1])

        self.world = args.fk_world
        self.pos_repr = args.pos_repr
        self.quater = args.rotation == 'quaternion'

    def forward_from_raw(self, raw, offset, world=None, quater=None):
        if world is None: world = self.world
        if quater is None: quater = self.quater
        if self.pos_repr == '3d':
            position = raw[:, -3:, :]
            rotation = raw[:, :-3, :]
        elif self.pos_repr == '4d':
            raise Exception('Not support')
        if quater:
            rotation = rotation.reshape((rotation.shape[0], -1, 4, rotation.shape[-1]))
            identity = torch.tensor((1, 0, 0, 0), dtype=torch.float, device=raw.device)
        else:
            rotation = rotation.reshape((rotation.shape[0], -1, 3, rotation.shape[-1]))
            identity = torch.zeros((3, ), dtype=torch.float, device=raw.device)
        identity = identity.reshape((1, 1, -1, 1))
        new_shape = list(rotation.shape)
        new_shape[1] += 1
        new_shape[2] = 1
        rotation_final = identity.repeat(new_shape)
        for i, j in enumerate(self.rotation_map):
            rotation_final[:, j, :, :] = rotation[:, i, :, :]
        return self.forward(rotation_final, position, offset, world=world, quater=quater)

    '''
    rotation should have shape batch_size * Joint_num * (3/4) * Time
    position should have shape batch_size * 3 * Time
    offset should have shape batch_size * Joint_num * 3
    output have shape batch_size * Time * Joint_num * 3
    '''
    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False, world=True):
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        result = torch.empty(rotation.shape[:-1] + (3, ), device=position.device)


        norm = torch.norm(rotation, dim=-1, keepdim=True)
        #norm[norm < 1e-10] = 1
        rotation = rotation / norm


        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.topology):
            if pi == -1:
                assert i == 0
                continue

            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
            result[..., i, :] = torch.matmul(transform[..., i, :, :], offset[..., i, :, :]).squeeze()
            if world: result[..., i, :] += result[..., pi, :]
        return result

    def from_local_to_world(self, res: torch.Tensor):
        res = res.clone()
        for i, pi in enumerate(self.topology):
            if pi == 0 or pi == -1:
                continue
            res[..., i, :] += res[..., pi, :]
        return res

    @staticmethod
    def transform_from_euler(rotation, order):
        rotation = rotation / 180 * math.pi
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
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


class InverseKinematics:
    def __init__(self, rotations: torch.Tensor, positions: torch.Tensor, offset, parents, constrains):
        self.rotations = rotations
        self.rotations.requires_grad_(True)
        self.position = positions
        self.position.requires_grad_(True)

        self.parents = parents
        self.offset = offset
        self.constrains = constrains

        self.optimizer = torch.optim.Adam([self.position, self.rotations], lr=1e-3, betas=(0.9, 0.999))
        self.crit = nn.MSELoss()

    def step(self):
        self.optimizer.zero_grad()
        glb = self.forward(self.rotations, self.position, self.offset, order='', quater=True, world=True)
        loss = self.crit(glb, self.constrains)
        loss.backward()
        self.optimizer.step()
        self.glb = glb
        return loss.item()

    def tloss(self, time):
        return self.crit(self.glb[time, :], self.constrains[time, :])

    def all_loss(self):
        res = [self.tloss(t).detach().numpy() for t in range(self.constrains.shape[0])]
        return np.array(res)

    '''
        rotation should have shape batch_size * Joint_num * (3/4) * Time
        position should have shape batch_size * 3 * Time
        offset should have shape batch_size * Joint_num * 3
        output have shape batch_size * Time * Joint_num * 3
    '''

    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False,
                world=True):
        '''
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        '''
        result = torch.empty(rotation.shape[:-1] + (3,), device=position.device)

        norm = torch.norm(rotation, dim=-1, keepdim=True)
        rotation = rotation / norm

        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.parents):
            if pi == -1:
                assert i == 0
                continue

            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze()
            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
            if world: result[..., i, :] += result[..., pi, :]
        return result

    @staticmethod
    def transform_from_euler(rotation, order):
        rotation = rotation / 180 * math.pi
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
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
