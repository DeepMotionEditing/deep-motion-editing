import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as pe
from matplotlib import cm
import torch
import argparse
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))

from py_utils import to_float


"""
Motion info: 
    joint parents, foot_idx
"""
J = 21
parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19])
joint_foot_indices = [3, 4, 7, 8]
joint_sizes = [3 for i in range(J)]
head_index = 12
joint_sizes[head_index] = 7

"""
Anim info:
    limb_colors
    joint_colors
    scale
    centered
"""

cmap = cm.get_cmap("Pastel2")
limb_colors = [cmap(x) for x in np.arange(0, 1, 0.125)]

cmap = cm.get_cmap("Set2")
joint_colors = [cmap(x) for x in np.arange(0, 1, 0.125)]

scale = 0.75
centered = True


def init_2d_plot(fig, subplot_pos, scale):
    ax = fig.add_subplot(subplot_pos)
    ax.set_xlim(-scale*40, scale*40)
    ax.set_ylim(-scale*40, scale*40)
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    return ax


def init_3d_plot(fig, subplot_pos, scale):
    ax = fig.add_subplot(subplot_pos, projection='3d') # This projection type determines the #axes
    rscale = scale * 20 # 15
    ax.set_xlim3d(-rscale, rscale)
    ax.set_zlim3d(-rscale, rscale)
    ax.set_ylim3d(-rscale, rscale)

    facec = (254, 254, 254)
    linec = (240, 240, 240)
    facec = list(np.array(facec) / 256.0) + [1.0]
    linec = list(np.array(linec) / 256.0) + [1.0]

    ax.w_zaxis.set_pane_color(facec)
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    X = np.arange(-20, 25, 5)
    Y = np.arange(-20, 25, 5)
    xlen = len(X)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape) - rscale # place it at a lower surface

    colortuple = (facec, linec)
    colors = np.zeros((Z.shape + (4, )))
    for y in range(ylen):
        for x in range(xlen):
            colors[y, x] = colortuple[(x + y) % len(colortuple)]

    # Plot the surface with face colors taken from the array we made.
    surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0., zorder=-1, shade=False)

    ax.w_zaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_color(linec)
    ax.w_xaxis.line.set_lw(0.)
    ax.w_xaxis.line.set_color(linec)

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])

    ax.view_init(20, -60) # -40 for the other direction

    return ax


def init_lines(ax, anim, dim, color=limb_colors[0], scale=1.0):
    init_pos = [[0, 0] for i in range(dim)]
    return [ax.plot(*init_pos, color=color, zorder=3,
                    linewidth=2 * scale, solid_capstyle='round',
                    path_effects=[pe.Stroke(linewidth=3 * scale, foreground='black'),
                                  pe.Normal()])[0] for _ in range(anim.shape[1])]


def init_dots(ax, anim, dim, color='white', scale=1.0):
    init_pos = [[0] for i in range(dim)]
    return [ax.plot(*init_pos, color=color, zorder=3,
                    linewidth=2, linestyle='',
                    marker="o", markersize=joint_sizes[i] * scale,
                    path_effects=[pe.Stroke(linewidth=1.5 * scale, foreground='black'), pe.Normal()]
                    )[0] for i in range(anim.shape[1])]


def _anim_skel(lines, dots, anim, dim, i):
    i = min(i, len(anim) - 1)
    if dim == 3:
        for j in range(len(parents)):
            if parents[j] != -1:
                lines[j].set_data(
                    [ anim[i, j, 0],  anim[i, parents[j], 0]],
                    [-anim[i, j, 2], -anim[i, parents[j], 2]])
                lines[j].set_3d_properties(
                    [ anim[i, j, 1],  anim[i, parents[j], 1]])

            dots[j].set_data([anim[i, j, 0]], [-anim[i, j, 2]])
            dots[j].set_3d_properties([anim[i, j, 1]])

    else:
        for j in range(len(parents)):
            if parents[j] != -1:
                lines[j].set_data(
                    [anim[i, j, 0], anim[i, parents[j], 0]],
                    [anim[i, j, 1], anim[i, parents[j], 1]])
            dots[j].set_data([anim[i, j, 0]], [anim[i, j, 1]])

    return [lines, dots]


def _anim_foot_contact(dots, foot_contact, i):
    i = min(i, len(foot_contact) - 1)
    for j, f_idx in enumerate(joint_foot_indices):
        color = 'red' if foot_contact[i, j] == 1.0 else 'blue'
        dots[f_idx].set_color(color)
    return [dots]


class Motion4Anim:
    def __init__(self, title, motion, foot, limb_color=limb_colors[0], joint_color=joint_colors[0]):
        self.title = title
        self.motion = motion
        if centered:
            self.motion -= self.motion[0:1, 0:1, :]
            self.motion = glb2centered(motion)
        self.T = motion.shape[0]
        self.dims = motion.shape[-1]
        self.foot = foot
        self.ax = None
        self.lines = None
        self.dots = None
        self.limb_color = limb_color
        self.joint_color = joint_color

    def set_anim(self, fig, pos, single=False):
        if self.dims == 2:
            self.ax = init_2d_plot(fig, pos, scale)
        else:
            self.ax = init_3d_plot(fig, pos, scale)
        if self.title is not None:
            self.ax.set_title(self.title)

        plot_scale = 2.0 if single else 1.0
        self.lines = init_lines(self.ax, self.motion, self.dims, self.limb_color, scale=plot_scale)
        self.dots = init_dots(self.ax, self.motion, self.dims, self.joint_color, scale=plot_scale)

    def anim_skel(self, i):
        return _anim_skel(self.lines, self.dots, self.motion, self.dims, i)

    def anim_foot_contact(self, i):
        if self.foot is not None:
            return _anim_foot_contact(self.dots, self.foot, i)
        else:
            return []

    def anim_i(self, i):
        return self.anim_skel(i) + self.anim_foot_contact(i)


def plot_motions(motions, size=4, interval=26.67, fps=10, save=False, save_path=None):
    """motions: list of Motion4Anim}"""
    if not isinstance(motions, list):
        motions = [motions]

    N = len(motions)
    T = 0
    for mt in motions:
        T = max(T, mt.T)

    fig = plt.figure(figsize=(N * size, size))
    init_pos = 100 + 10 * N + 1

    for i, mt in enumerate(motions):
        mt.set_anim(fig, init_pos + i)

    def animate(i):
        changed = []
        for mt in motions:
            changed += mt.anim_i(i)
        return changed

    plt.tight_layout()
    ani = animation.FuncAnimation(fig, animate, np.arange(T), interval=interval)

    if save:
        assert save_path is not None, "save_path is None!"
        print(f'Start saving motion to {save_path}')
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if not save_path.endswith('.mp4'):
            save_path += '.mp4'
        ani.save(save_path, writer='ffmpeg', fps=fps)
        print(f'Motion saved to {save_path}')
    else:
        plt.show()
        opt = input("save? Yes/No/Exit")
        return opt


def glb2centered(glb):
    """
    input: positions - glb [T, J, (3/2)] -- single clip!
    output: motion with average root (x(, z)) = (0(, 0))
    """
    root_avg = np.mean(glb[:, 0:1, :], axis=0, keepdims=True)
    root_avg[0, 0, 1] = 0  # y shouldn't change
    return glb - root_avg


def rotate_motion(mt):
    def rotate_motion3d(mt):
        if mt[-1, 0, 0] - mt[0, 0, 0] < 0:
            mt[..., 0] = -mt[..., 0]
        if mt[-1, 0, 2] - mt[0, 0, 2] < 0:
            mt[..., 2] = -mt[..., 2]
        if mt[-1, 0, 0] - mt[0, 0, 0] < mt[-1, 0, 2] - mt[0, 0, 2]: # move in z dir
            tmp = mt[..., 0].copy()
            mt[..., 0] = mt[..., 2].copy()
            mt[..., 2] = tmp
        return mt

    def rotate_motion2d(mt):
        """
        if mt[-1, 0, 0] > mt[0, 0, 0]:
            mt[..., 0] = -mt[..., 0]
        """
        return mt

    if mt.shape[-1] == 2:
        return rotate_motion2d(mt)
    elif mt.shape[-1] == 3:
        return rotate_motion3d(mt)
    else:
        assert 0, "motion dimension is {mt.shape[-1]}"


def visualize(data, save=False, save_path=None):
    """data: dict {title: {motion:xxx, foot_contact:xxx}}"""

    motions = []

    for i, (title, motion_dict) in enumerate(data.items()):
        motion = to_float(motion_dict['motion']).copy()
        motion = rotate_motion(motion)  # [T, J, 2/3]
        foot_contact = motion_dict['foot_contact']  # [T, 4]

        motions.append(Motion4Anim(title,
                                   motion,
                                   foot_contact,
                                   limb_colors[i],
                                   joint_colors[i]
                                   ))
    plot_motions(motions, save=save, save_path=save_path)


def to_numpy(data):
    output = []
    for d in data:
        if isinstance(d, torch.Tensor):
            output.append(d.detach().numpy())
        else:
            output.append(d)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    return args


def load_output(filename):
    data = torch.load(filename, map_location='cpu')
    print(list(data.keys()))
    return data


def main(args):
    from utils.animation_data import AnimationData
    from utils.animation_2d_data import AnimationData2D
    data = load_output(args.file)
    total = len(data["trans"])
    content, style, foot_contact, trans, recon = data["content"], data["style"], data["foot_contact"], data["trans"], data["recon"]
    content_meta, style_meta = data["content_meta"], data["style_meta"]
    selected = list(range(total))
    print(total)
    # for test, selected = [6, 12, 7, 11, 4]
    for i in selected:

        if style_meta[i] == 0:
            style_meta[i] = {"style": [str(i)]}
        if content_meta[i] == 0:
            content_meta[i] = {"style": [str(i)]}

        vis_dict = {}
        cur_foot_contact = foot_contact[i].transpose(1, 0)
        if style[i].shape[0] == content[i].shape[0]:  # 3d
            cur_style = AnimationData.from_network_output(to_float(style[i])).get_global_positions()
        else:  # 2d
            cur_style = AnimationData2D.from_style2d(to_float(style[i])).get_projection()
        raws = [trans[i], recon[i], content[i]]
        cur_trans, cur_recon, cur_content = [AnimationData.from_network_output(to_float(raw)).get_global_positions() for raw in raws]
        vis_dict[" ".join(("style", style_meta[i]["style"][0]))] = {"motion": cur_style, "foot_contact": None}
        vis_dict["trans"] = {"motion": cur_trans, "foot_contact": cur_foot_contact}
        vis_dict["recon"] = {"motion": cur_recon, "foot_contact": cur_foot_contact}
        vis_dict[" ".join(("content", content_meta[i]["style"][0]))] = {"motion": cur_content, "foot_contact": cur_foot_contact}

        visualize(vis_dict)


if __name__ == "__main__":
    args = parse_args()
    main(args)
