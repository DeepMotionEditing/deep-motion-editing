import sys
import torch
from models.Kinematics import InverseKinematics
from datasets.bvh_parser import BVH_file
from tqdm import tqdm

sys.path.append('../utils')

import BVH as BVH
import Animation as Animation
from Quaternions_old import Quaternions


L = 5  # #frame to look forward/backward


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


def get_character_height(file_name):
    file = BVH_file(file_name)
    return file.get_height()


def get_foot_contact(file_name, ref_height):
    anim, names, _ = BVH.load(file_name)

    ee_ids = get_ee_id_by_names(names)

    glb = Animation.positions_global(anim)  # [T, J, 3]

    ee_pos = glb[:, ee_ids, :]
    ee_velo = ee_pos[1:, ...] - ee_pos[:-1, ...]
    ee_velo = torch.tensor(ee_velo) / ref_height
    ee_velo_norm = torch.norm(ee_velo, dim=-1)
    contact = ee_velo_norm < 0.003
    contact = contact.int()
    padding = torch.zeros_like(contact)
    contact = torch.cat([padding[:1, :], contact], dim=0)
    return contact.numpy()


def get_ee_id_by_names(joint_names):
    ees = ['RightToeBase', 'LeftToeBase', 'LeftFoot', 'RightFoot']
    ee_id = []
    for i, ee in enumerate(ees):
        ee_id.append(joint_names.index(ee))
    return ee_id


def fix_foot_contact(input_file, foot_file, output_file, ref_height):
    anim, name, ftime = BVH.load(input_file)

    fid = get_ee_id_by_names(name)
    contact = get_foot_contact(foot_file, ref_height)

    glb = Animation.positions_global(anim)  # [T, J, 3]

    T = glb.shape[0]

    for i, fidx in enumerate(fid):  # fidx: index of the foot joint
        fixed = contact[:, i]  # [T]
        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()
            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(L):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(L):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break
            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    # glb is ready

    anim = anim.copy()

    rot = torch.tensor(anim.rotations.qs, dtype=torch.float)
    pos = torch.tensor(anim.positions[:, 0, :], dtype=torch.float)
    offset = torch.tensor(anim.offsets, dtype=torch.float)

    glb = torch.tensor(glb, dtype=torch.float)

    ik_solver = InverseKinematics(rot, pos, offset, anim.parents, glb)

    print('Fixing foot contact using IK...')
    for i in tqdm(range(50)):
        ik_solver.step()

    rotations = ik_solver.rotations.detach()
    norm = torch.norm(rotations, dim=-1, keepdim=True)
    rotations /= norm

    anim.rotations = Quaternions(rotations.numpy())
    anim.positions[:, 0, :] = ik_solver.position.detach().numpy()

    BVH.save(output_file, anim, name, ftime)
