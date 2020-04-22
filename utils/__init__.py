import sys
import os
import numpy as np
import scipy.ndimage.filters as filters
import torch
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)

import BVH
import Animation
from Quaternions import Quaternions
from Pivots import Pivots
from JointsConfig import needed_joints

class Skel:
    def __init__(self, filename=pjoin(BASEPATH, "../global_info/FK_skel")):
        data = torch.load(filename)
        offset = data["offsets"]
        topology = data["parents"]
        if isinstance(offset, torch.Tensor):
            offset = offset.numpy()
        self.offset = offset
        self.topology = topology

skel = Skel()

def _rot_to_pos(rotations, rtpos=None, skel=skel, trim=True):
    """
    input: rotations [(B, ) T, J, 4], rtpos [(B, ) T, 3]
    output: pos [(B, ) T, J, 3]
    """
    norm = np.sqrt(np.sum(rotations ** 2, axis=-1, keepdims=True))
    norm = 1.0 / norm
    rotations *= norm
    # rotations = rotations / norm
    transforms = Quaternions(rotations).transforms()  # [..., J, 3, 3]

    glb = np.zeros(rotations.shape[:-1] + (3, ))  # [..., J, 3]
    if rtpos is not None:
        glb[..., 0, :] = rtpos
    for i, pi in enumerate(skel.topology):
        if pi == -1:
            assert i == 0
            continue
        glb[..., i, :] = np.matmul(transforms[..., pi, :, :],
                                   skel.offset[i])
        glb[..., i, :] += glb[..., pi, :]
        transforms[..., i, :, :] = np.matmul(transforms[..., pi, :, :],
                                             transforms[..., i, :, :])
    if trim:
        return glb[..., needed_joints, :]
    else:
        return glb


def rot_to_glb(rot, skel=skel, trim=True):
    """
    input: rotations [(B, ) J * 4 + 3, T]
    output: global positions [(B, ) T, J, 3]
    """
    rot = np.swapaxes(rot, -1, -2)
    rotations, rt_positions = rot[..., :-3], rot[..., -3:] # [B, T, J * 4], [B, T, 3]
    rotations = rotations.reshape(rotations.shape[:-1] + (-1, 4))  # [B, T, J, 4]
    return _rot_to_pos(rotations, rtpos=rt_positions, trim=trim)

def nrot_to_glb(rot, skel=skel):
    """
    input: new! rotations [(B, ) J * 4 + 3 + 1, T]
    output: global positions [(B, ) T, J, 3]
    """
    rot = np.swapaxes(rot, -1, -2)
    rotations, rt_positions, rt_rotations = rot[..., :-4], rot[..., -4:-1], rot[..., -1:]
    rotations = rotations.reshape(rotations.shape[:-1] + (-1, 4))  # [..., J, 4]
    norm = np.sqrt(np.sum(rotations ** 2, axis=-1,keepdims=True))
    norm = 1.0 / norm
    rotations *= norm
    # rotations = rotations / norm

    rt_rotations = Quaternions(np.array(Pivots(rt_rotations).quaternions()))
    ori_rt_rotations = Quaternions(rotations)[:, 0:1]
    ori_rt_rotations = rt_rotations * ori_rt_rotations
    rotations[:, 0:1, :] = np.array(ori_rt_rotations) # [T, J, 4]

    return _rot_to_pos(rotations, rtpos=rt_positions)

def nrot_to_glb_and_rot(rot, skel=skel):
    """
    input: new! rotations [(B, ) J * 4 + 3 + 1, T]
    output: global positions [(B, ) T, J, 3]
    """
    rot = np.swapaxes(rot, -1, -2)
    rotations, rt_positions, rt_rotations = rot[..., :-4], rot[..., -4:-1], rot[..., -1:]
    rotations = rotations.reshape(rotations.shape[:-1] + (-1, 4))  # [..., J, 4]
    norm = np.sqrt(np.sum(rotations ** 2, axis=-1,keepdims=True))
    rotations = rotations / norm

    rt_rotations = Quaternions(np.array(Pivots(rt_rotations).quaternions()))
    ori_rt_rotations = Quaternions(rotations)[:, 0:1]
    ori_rt_rotations = rt_rotations * ori_rt_rotations
    rotations[:, 0:1, :] = np.array(ori_rt_rotations) # [T, J, 4]

    return _rot_to_pos(rotations, rtpos=rt_positions, trim=False), rotations

def nrot_for_forwardBVH(rot, skel=skel):
    """
    input: new! rotations [J * 4 + 3 + 1, T]
    output: global positions [T, J, 3] & rotation -> facing z+
    """
    rot = np.swapaxes(rot, -1, -2)
    rotations, rt_positions, rt_rotations = rot[..., :-4], rot[..., -4:-1], rot[..., -1:]

    rotations = rotations.reshape(rotations.shape[:-1] + (-1, 4))  # [..., J, 4]
    norm = np.sqrt(np.sum(rotations ** 2, axis=-1,keepdims=True))
    rotations = rotations / norm

    """
    # based on distance
    # 0: z+, 1: x+, 2: z-, 3: x-,
    trail = np.zeros(4)
    trail[1] = rt_positions[-1, 0] - rt_positions[0, 0]
    trail[3] = -trail[1]
    trail[0] = rt_positions[-1, 2] - rt_positions[0, 2]
    trail[2] = -trail[0]

    dir = np.argmax(trail)
    """

    # based on forward direction in rt_rotations
    # 0: z+, 1: x+, 2: z-, 3: x-,
    trail = np.zeros(4)
    for dir in range(4):
        tmp = rt_rotations - dir * np.pi * 0.5 * np.ones(rt_rotations.shape)
        tmp = np.abs(tmp)
        tmp = np.minimum(tmp, 2.0 * np.pi * np.ones(rt_rotations.shape) - tmp)
        trail[dir] = np.sum(tmp)

    dir = np.argmin(trail)


    rt_rotations = rt_rotations - dir * np.pi * 0.5 * np.ones(rt_rotations.shape)

    # np.set_printoptions(precision=3, suppress=True)

    # print(rt_positions)

    for d in range(dir):
        tmp = rt_positions[..., 0].copy()
        rt_positions[..., 0] = -rt_positions[..., 2].copy()
        rt_positions[..., 2] = tmp

    # print(rt_positions)

    rt_rotations = Quaternions(np.array(Pivots(rt_rotations).quaternions()))
    ori_rt_rotations = Quaternions(rotations)[:, 0:1]
    ori_rt_rotations = rt_rotations * ori_rt_rotations
    rotations[:, 0:1, :] = np.array(ori_rt_rotations) # [T, J, 4]

    return _rot_to_pos(rotations, rtpos=rt_positions, trim=False), rotations




def glb_to_3dpos(glb): # rt last!!
    rt_positions = glb[..., 0:1, :].copy() # [(B, ) T, 1, 3]
    glb -= rt_positions # relative positions
    pos = np.concatenate([glb[..., 1:, :], rt_positions], axis=-2)  # [(B, ) T, J - 1, 3]
    pos = pos.reshape(pos.shape[:-2] + (-1, )) # [(B, ) T, J * 3]
    pos = np.swapaxes(pos, -1, -2) # [(B, ) J * 3, T]
    return pos


def pos3d_to_glb(pos):
    """
    input: pos [(B,) (J - 1) * 3 + 3, T]
    output: positions [(B,) T, J, 3]
    """
    pos = pos.swapaxes(-1, -2) # [B, T, J * 3]
    pos = pos.reshape(pos.shape[:-1] + (-1, 3))
    rt, joints = pos[..., -1:, :], pos[..., :-1, :]
    joints += rt
    return np.concatenate([rt, joints], axis=-2)



def rot_to_3dpos(rot, skel=skel):
    """
    input: rotations [(B, ) J * 4 + 3, T]
    output: 3d pos (rt position + relative pos) for network input [(B, ) J * 3, T]
    """
    glb = rot_to_glb(rot, skel)  # [(B, ) T, J, 3]
    return glb_to_3dpos(glb)

def nrot_to_rfree(nrot, skel=skel):
    """
    input: new! rotations [(B, ) J * 4 + 4, T]
    output: rfree [(B, ) (J - 1) * 3 + 4, T]
    """
    rot, rt = nrot[..., :-4, :], nrot[..., -4:, :] # [(B, ) J * 4, T], [B, 4, T]
    rot = rot.swapaxes(-1, -2) # [(B, ), T, J * 4]
    rot = rot.reshape(rot.shape[:-1] + (-1, 4))  # [(B, ) T, J, 4]
    pos = _rot_to_pos(rot, skel=skel) # [(B, ) T, J, 3]
    pos = pos[..., 1:, :]  # take rt out
    pos = pos.reshape(pos.shape[:-2] + (-1,)) # [(B, ) T, J * 3]
    pos = pos.swapaxes(-1, -2) # [(B, ), J * 3, T]
    return np.concatenate([pos, rt], axis=-2) # [(B, ) J * 3 + 4, T]

def across_from_glb(positions):
    """
    input: positions [(B, )T, J, 3]
    output: across for each frame [(B, ), T, 3]
    """
    hip_l, hip_r, sdr_l, sdr_r = 2, 6, 14, 18
    across = positions[..., hip_l, :] - positions[..., hip_r, :] + \
             positions[..., sdr_l, :] - positions[..., sdr_r, :]  # [T, 3]

    # length will be of shape [..., J, 1]
    norm = np.sqrt((across ** 2).sum(axis=-1))[...,np.newaxis]
    norm = 1.0 / norm
    across *= norm
    # across = across / np.sqrt((across ** 2).sum(axis=-1))[...,np.newaxis]

    return across

def glb_to_proj(glb):
    """
    input: positions [(B,) T, J, 2]
    output: proj [(B,) (J - 1) * 2 + 2, T]
    """
    rt = glb[..., 0:1, :].copy()
    glb -= rt
    glb = np.concatenate([glb[..., 1:, :], rt], axis=-2)
    glb = glb.reshape(glb.shape[:-2] + (-1, ))
    glb = glb.swapaxes(-1, -2)
    return glb


def proj_to_glb(proj):
    """
    input: proj [(B,) (J - 1) * 2 + 2, T]
    output: positions [(B,) T, J, 2]
    """
    proj = proj.swapaxes(-1, -2) # [B, T, J * 2]
    proj = proj.reshape(proj.shape[:-1] + (-1, 2))
    rt, joints = proj[..., -1:, :], proj[..., :-1, :]
    joints += rt
    return np.concatenate([rt, joints], axis=-2)

def rfree_to_glb(rfree):
    """
    input: rfree [(B, ) (J - 1) * 3 + 3 + 1, T]
    output: positions [(B, ) T, J, 3]
    """

    rfree = np.swapaxes(rfree, -1, -2) # to [B, T, J * xxx]
    positions, rt_positions, rt_rotations = rfree[..., :-4], rfree[..., -4:-1], rfree[..., -1:]

    positions = np.reshape(positions, positions.shape[:-1] + (-1, 3))

    rt_rotations = Pivots(rt_rotations).quaternions()
    positions = rt_rotations * positions

    rt_positions = np.reshape(rt_positions, rt_positions.shape[:-1] + (-1, 3))
    positions += rt_positions
    positions = np.concatenate([rt_positions, positions], axis=-2)

    return positions

def ft_from_glb(positions):

    fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
    velfactor = np.array([0.05, 0.05])
    feet_contact = []
    for fid_index in [fid_l, fid_r]:
        foot_vel = (positions[1:, fid_index] - positions[:-1, fid_index]) ** 2  # [T - 1, 2, 3]
        foot_vel = np.sum(foot_vel, axis=-1)  # [T - 1, 2]
        foot_contact = (foot_vel < velfactor).astype(np.float)
        feet_contact.append(foot_contact)
    feet_contact = np.concatenate(feet_contact, axis=-1)  # [T - 1, 4]

    return feet_contact # [T - 1, 4]!

def phase_from_ft(foot_contact):
    """
    foot_contact: [T - 1, 4]
    take the 3(0) & 7(2) as standards
    first 1 in 3: 1, first 1 in 7: -1
    """
    num_circles = 0
    circle_length = 0
    total_length = len(foot_contact)

    ft = foot_contact[:, [0, 2]].astype(np.int)
    ft_start = np.zeros(ft.shape)
    phases = np.zeros((ft.shape[0], 1))

    """
    calculate the average "half circle"
    find the first and last "01" pattern
    """
    for j in range(2):
        for i in range(1, total_length):
            ft_start[i, j] = (ft[i - 1, j] == 0 and ft[i, j] == 1)
        print(ft_start[:, j])

    last = -1
    beg_i, end_i = -1, -1
    starts = []


    for i in range(total_length):
        if ft_start[i, 0] or ft_start[i, 1]:
            if last != -1:
                num_circles += 1
                circle_length += i - last
            else:
                beg_i = i
            last = i
            end_i = i
            starts.append(i)


    avg_circle = 0 if num_circles == 0 else circle_length * 1.0 / num_circles
    print("%d circles, total length = %d, avg length = %.3lf" % (num_circles, circle_length, avg_circle))

    # prev, next

    # [0, beg_i - 1]: first incomplete circle

    prev_pos = min(0, beg_i - avg_circle)
    prev_val = 0 if ft_start[beg_i, 1] == 1 else 1  #  0 if next step is on the right
    cir_i = 0
    next_pos = starts[cir_i]

    for i in range(total_length):
        if i == next_pos:
            prev_pos = next_pos
            prev_val = 1 - prev_val
            cir_i += 1
            if cir_i == num_circles + 1:
                next_pos = max(total_length + 1, next_pos + avg_circle)
            else:
                next_pos = starts[cir_i]
        phases[i] = prev_val + (i - prev_pos) * 1.0 / (next_pos - prev_pos)

    print(phases)

    return phases

def yrot_from_glb(positions):
    """
    input: positions [(B, )T, J, 3]
    output: y-axis rotations in quaternions & pivots: [(B, ) T, 1, 4], [(B, )T, 1]
    quaters to apply to motions, pivots for appending to the representation
    """

    """ Remove root translations """
    rt_positions = positions[..., 0:1, :].copy()  # [..., 1, 3]
    positions -= rt_positions # relative positions, [..., J, 3]
    rt_positions -= rt_positions[0, :, :].copy()  # set init pos to zero

    """ Get forward direction """
    # as long as the joints are all present (including root),
    # it doesn't matter whether the positions passed are relative or not
    across = across_from_glb(positions)

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Remove Y Rotation """
    target = np.tile(np.array([0,0,1]), forward.shape[:-1] + (1, ))
    rotations = Quaternions.between(forward, target)[..., np.newaxis, :] # from [B, T, 4] to [B, T, 1, 4]

    # -rotation corresponds to the "global rotations"
    rt_rotations = Pivots.from_quaternions(-rotations).ps # [...T, 1]

    return rotations, rt_rotations


def glb_to_rfree(positions):
    """
    input: positions [(B, )T, J, 3]
    output: rfree [(B, ), (J - 1) * 3 + 3 + 1, T]
    """

    """ Remove root translations """
    rt_positions = positions[..., 0:1, :].copy()  # [..., 1, 3]
    positions -= rt_positions # relative positions, [..., J, 3]
    rt_positions -= rt_positions[0, :, :].copy()  # set init pos to zero

    """ Get forward direction """
    # as long as the joints are all present (including root),
    # it doesn't matter whether the positions passed are relative or not
    across = across_from_glb(positions)

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]


    """ Remove Y Rotation """
    target = np.tile(np.array([0,0,1]), forward.shape[:-1] + (1, ))
    rotation = Quaternions.between(forward, target)[..., np.newaxis, :] # from [B, T, 4] to [B, T, 1, 4]
    # apply rotation to positions (after removing translations)
    positions = rotation * positions

    # -rotation corresponds to the "global rotations"
    rt_rotations = Pivots.from_quaternions(-rotation).ps # [...T, 1]

    T = positions.shape[-3]  # [(B, ), T, J, 3]
    # exclude root joint, convert to [(B, ), T, (J - 1) * 3]
    positions = positions[..., 1:, :].reshape(positions.shape[:-3] + (T, -1))
    rt_positions = rt_positions.reshape(rt_positions.shape[:-3] + (T, -1))
    rfree = np.concatenate([positions, rt_positions, rt_rotations], axis=-1)
    rfree = np.swapaxes(rfree, -1, -2)

    return rfree

def glb2centered(glb):
    """
    input: positions - glb [T, J, (3/2)] -- single clip!
    output: motion with average root (x(, z)) = (0(, 0))
    """
    root_avg = np.mean(glb[:, 0:1, :], axis=0, keepdims=True)
    root_avg[0, 0, 1] = 0  # y shouldn't change
    return glb - root_avg

def glb2static(glb):
    root_pos = glb[:, 0:1, :].copy()
    root_pos[:, :, 1] = 0 # set y to zero
    return glb - root_pos

def add_glb_velocity(pos, vel):
    dims = pos.shape[-1]
    vel = np.reshape(vel, (1, -1))
    cur = np.zeros((1, dims))
    output = pos.copy()
    for i in range(len(pos)):
        output[i] += cur.copy()
        cur += vel
    return output

view_angles = [(0, -np.pi / 2, 0),
               (0, -np.pi / 3, 0),
               (0, -np.pi / 6, 0),
               (0, 0, 0),
               (0, np.pi / 6, 0),
               (0, np.pi / 3, 0),
               (0, np.pi / 2, 0)]


def rotate_coordinates(local3d, angles):
    """
    Rotate local rectangular coordinates from given view_angles.

    :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
    :param angles: tuple of length 3. Rotation angles around each GLOBAL axis.
    :return:

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

def get_projections(glb, view_angles):
    """
    rfree: [(B,) (J - 1) * 3 + 3 + 1, T]
    glb: [(B, ) T, J, 3] global positions
    view_angles: [-pi/2, etc.]
    output: projections [(B, ), len(view_angles), (J - 1) * 2 + 2, T]
    """
    # glb = rfree_to_glb(rfree)  # [(B,) T, J, 3]

    # get local_x [(B,) 3]
    local_x = across_from_glb(glb)  # [(B,) T, 3]
    local_x = local_x.mean(axis=-2) # [(B, ) 3]
    local_x = local_x * (1.0 / np.sqrt(np.sum(local_x ** 2, axis=-1, keepdims=True)))

    # convert glb to [(B,) T, J - 1 + 1, 3], local joint positions + global root positions
    rt = glb[..., 0:1, :].copy()  # [(B,) T, 1, 3]
    joints = glb - rt
    rt -= rt[..., 0:1, :, :].copy() # remove the first frame
    joints = np.concatenate([joints[..., 1:, :], rt], axis=-2) # [(B,), T, J - 1, 3], [(B,) T, 1, 3]

    ret = []
    for view_angle in view_angles:
        ret.append(motion_projection(joints, local_x, view_angle)) #  [(B,) J * 2, T]
    ret = np.stack(ret, axis=-3)  # [(B,) V, J * 2, T]
    return ret

def get_rfree_projections(rfree, view_angles):
    return get_projections(rfree_to_glb(rfree), view_angles)

def get_rot_projections(rot, view_angles):
    return get_projections(rot_to_glb(rot), view_angles)

def get_nrot_projections(nrot, view_angles):
    return get_projections(nrot_to_glb(nrot), view_angles)

def motion_projection(motion, local_x, view_angle=None):
    """
    :param motion: motion in relative joint positions & global root positions
                   [(B,) T, (J - 1) + 1, 3]
    :param local_x: [(B,) 3], local x-axis
    :param view_angle: [3], the angles to rotate
    :return: motion_proj [(B,) J * 2, T]
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


def normalize_motion(motion, mean_pose, std_pose):
    """
    inputs:
    motion: (V, C, T) or (C, T)
    mean_pose: (C)
    std_pose: (C)
    """
    return (motion - mean_pose[:, np.newaxis]) / std_pose[:, np.newaxis]


def normalize_motion_inv(motion, mean_pose, std_pose):
    """
    :param motion: (?, J * 2, T) or (J * 2, T)
    :return:
    """
    return motion * std_pose[:, np.newaxis] + mean_pose[:, np.newaxis]




def test_local3d():
    lx = np.array([4.0, 0.0, 3.0])
    ly = np.array([3.0, 0.0, 4.0])
    blx = np.array([lx, ly])
    get_local3d(lx, view_angles[1])
    get_local3d(ly, view_angles[1])
    get_local3d(blx, view_angles[1])

# test_local3d()
def test():
    la = np.random.rand(10, 21, 3)
    la[:, 0, :] = 0
    lb = np.random.rand(10, 21, 3)
    lb[:, 0, :] = 0
    bla = np.stack((la, lb), axis=0)

    rla = glb_to_rfree(la.copy())
    rlb = glb_to_rfree(lb.copy())
    rbla = glb_to_rfree(bla.copy())

    """
    x = rfree_to_glb(rla)
    y = rfree_to_glb(rbla)

    dis = np.sum((x - la) ** 2)
    print(dis)
    """

    pa = get_projections(rla, view_angles[0:2])
    pb = get_projections(rlb, view_angles[0:2])
    pc = get_projections(rbla, view_angles[0:2])
    print("=========================")
    print(np.mean(pa - pc[0]))
    print(np.mean(pb - pc[1]))


# test()

def test_rot_data():
    database = np.load("../data_ver6.npz", allow_pickle=True)["train"]
    motion, label = database[0]
    print(motion.shape)
    print(label)
    rot_to_glb(motion)

# test_rot_data()
