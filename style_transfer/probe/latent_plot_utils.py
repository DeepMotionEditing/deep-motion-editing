import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import tikzplotlib
from os.path import join as pjoin
BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from py_utils import ensure_dirs


def distinct_labels_and_indices(labels):
    distinct_labels = list(set(labels))
    distinct_labels.sort()
    num_labels = len(distinct_labels)
    indices_i = {label: [] for label in distinct_labels}
    for i, label in enumerate(labels):
        indices_i[label].append(i)
    indices_i = {label: np.array(indices) for label, indices in indices_i.items()}
    return num_labels, distinct_labels, indices_i


def plot2D(data, labels, title):
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - x_min) / (x_max - x_min)

    fig, ax = plt.subplots(figsize=(8, 8))
    cjet = cm.get_cmap("jet")

    num_labels, distinct_labels, indices = distinct_labels_and_indices(labels)

    for i, label in enumerate(distinct_labels):
        index = indices[label]
        ax.scatter(data[index, 0], data[index, 1], label=label, c=[cjet(1.0 * i / num_labels)], linewidths=0.)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0, 1, 1),
              title=title.split('/')[-1])

    fig.tight_layout()
    tikzplotlib.save("%s.tex" % title, figure=fig, strict=True)
    plt.savefig("%s.png" % title)

    return fig

def plot2D_overlay(data_list, labels_list, alpha_list, title):

    x_min, x_max = np.array((1e9, 1e9)), np.array((-1e9, -1e9))
    for data in data_list:
        x_min = np.minimum(x_min, np.min(data, axis=0))
        x_max = np.maximum(x_max, np.max(data, axis=0))

    for i in range(len(data_list)):
        data_list[i] = (data_list[i] - x_min) / (x_max - x_min)

    fig, ax = plt.subplots(figsize=(8, 8))
    cjet = cm.get_cmap("jet")

    indices_list = []
    distinct_labels = []
    for labels in labels_list:
        _, cur_labels, indices = distinct_labels_and_indices(labels)
        indices_list.append(indices)
        for label in cur_labels:
            if label not in distinct_labels:
                distinct_labels.append(label)
    num_labels = len(distinct_labels)

    for i, label in enumerate(distinct_labels):
        res = 0.0
        for data, labels, indices, alpha in zip(data_list, labels_list, indices_list, alpha_list):
            if label in indices.keys():
                index = indices[label]
            else:
                index = np.array([])
            c = cjet((1.0 * i + res) / (num_labels + 1))
            ax.scatter(data[index, 0], data[index, 1], label=label, c=[c], alpha=alpha, linewidths=0.)
            res += 0.3

    handles, labels = ax.get_legend_handles_labels()

    paired_handles = []
    handles_tot = len(handles) // 2
    for i in range(handles_tot):
        paired_handles.append((handles[i * 2], handles[i * 2 + 1]))

    ax.legend(handles=paired_handles, labels=distinct_labels, numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="center left", bbox_to_anchor=(1, 0, 1, 1),
              title=title.split('/')[-1])
    fig.tight_layout()
    tikzplotlib.save("%s.tex" % title, figure=fig, strict=True)
    plt.savefig("%s.png" % title)
    return fig


def plot2D_phase(data, labels, title):
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - x_min) / (x_max - x_min)

    figsize = (8, 8)
    add_width = 2
    new_width = figsize[0] + add_width

    fig = plt.figure(figsize=(new_width, figsize[1]))

    fac_l, fac_r = figsize[0] / new_width, add_width / new_width

    rect_l = [0.1, 0.1, 0.8, 0.8]
    rect_r = [0., 0.1, 0.2, 0.8]

    ax = fig.add_axes(np.array(rect_l) * np.array([fac_l, 1, fac_l, 1]))
    cax = fig.add_axes(np.array(rect_r) * np.array([fac_r, 1, fac_r, 1]) + np.array([fac_l, 0, 0, 0]))

    sin_labels = list(map(lambda l: np.sin(float(l)), labels))

    bla = ax.scatter(data[:, 0], data[:, 1], c=sin_labels, cmap="jet", alpha=1.0)

    # plt.colorbar(bla, cax=cax) <- some problem with the color bar..
    # fig.tight_layout()
    tikzplotlib.save("%s.tex" % title, figure=fig, strict=True)
    plt.savefig("%s.png" % title)
    return fig


tsne = None
def calc_tsne(raw):
    global tsne
    if tsne is None:
        tsne = TSNE(n_components=2, init='pca', random_state=7)  # n_iter = xxx
    result = tsne.fit_transform(raw)
    return result

pca = None
def calc_pca(raw):
    global pca
    if pca is None:
        pca = PCA(n_components=2)
    return pca.fit_transform(raw)

def calc_pca_curve(raw):
    pcan = PCA()
    pcan.fit_transform(raw)
    pct = pcan.explained_variance_ratio_
    prefix = np.cumsum(pct / np.sum(pct))

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.plot(list(range(1, 6)), prefix[:5])
    ax.plot(2, prefix[1], "ro")
    ax.annotate("{:.3f}% of variation".format(prefix[1] * 100),
                (2, prefix[1]),
                textcoords="offset points",
                xytext=(60, -20),
                ha="center")
    ax.set_xticks(list(range(1, 6)))
    ax.set_yticks(list(np.arange(0.5, 1.01, 0.1)))

    ax.set_xlabel("number of components")
    ax.set_ylabel("explained variance ratio")

    name = "pca_curve"
    tikzplotlib.save(name + ".tex", figure=fig, strict=True)
    plt.savefig("pca_curve.png")
    return pct


def plot_tsne(raw, labels, title):
    result = calc_tsne(raw)
    return plot2D(result, labels, title)


def plot_content_tsne(raw, slabels, clabels, title):
    name = title + "_tsne"
    path = name + ".npz"
    if os.path.exists(path):
        print("%s already exists" % path)
        result = np.load(path, allow_pickle=True)["result"]
    else:
        print("start to produce %s" % path)
        result = calc_tsne(raw)
        np.savez_compressed(name, result=result)
    plot2D(result, slabels, title + "_style_labels")
    plot2D(result, clabels, title + "_content_labels")


def calc_many_blas(raws, calc_single):
    lens = list(map(lambda x: len(x), raws))
    whole = np.concatenate(raws, axis=0)
    proj = calc_single(whole)
    ret = ()
    suml = 0
    for l in lens:
        ret += (proj[suml: suml + l],)
        suml += l
    return ret


def get_all_plots(data, output_path, writers, iter, summary=True,
                  style_cluster_protocols=('pca'),
                  separate_compute=False):
    """
    data: {"train": dict_train, "test": dict_test}
    dict_train: {"style2d_code": blabla, etc.}
    separate_compute: compute t-SNE for 2D & 3D separately
    """
    ensure_dirs(output_path)

    def fig_title(title):
        return pjoin(output_path, title)

    def add_fig(fig, title, phase):
        if summary:
            writers[phase].add_figure(title, fig, global_step=iter)

    keys = data["train"].keys()
    has2d = "style2d_code" in keys
    has3d = "style3d_code" in keys

    # style codes & adain params
    for suffix in ["_code", "_adain"]:

        codes_raw = []
        titles = []
        phases = []

        data_keys = []
        if has2d: data_keys.append("style2d" + suffix)
        if has3d: data_keys.append("style3d" + suffix)
        for key in data_keys:
            for phase in ["train", "test"]:
                codes_raw.append(data[phase][key])
                titles.append(f'{phase}_{key}')
                phases.append(phase)

        # calc tsne with style2/3d, train/test altogether
        for name, protocol in zip(['pca', 'tsne'], [calc_pca, calc_tsne]):
            if name not in style_cluster_protocols:
                continue
            style_codes = calc_many_blas(codes_raw, protocol)
            fig = plot2D_overlay([style_codes[0], style_codes[2]],
                                 [data["train"]["meta"]["style"], data["train"]["meta"]["style"]],
                                 [1.0, 0.5],
                                 fig_title(f'joint_embedding_{name}{suffix}'))
            add_fig(fig, f'joint_embedding_{name}{suffix}', "train")

            for i, (code, phase, title) in enumerate(zip(style_codes, phases, titles)):
                if separate_compute:
                    code = protocol(codes_raw[i])
                for label_type in ["style", "content"]:
                    fig = plot2D(code, data[phase]["meta"][label_type], fig_title(f'{title}_{name}_{label_type}'))
                    add_fig(fig, f'{title}_{name}_{label_type}', phase)


    # content codes (train only)
    content_code_pca = calc_pca(data["train"]["content_code"])

    for label in ["style", "content", "phase"]:
        if label == "phase":
            indices = [i for i in range(len(data["train"]["meta"]["content"])) if data["train"]["meta"]["content"][i] == "walk"]
            walk_code = content_code_pca[np.array(indices)]
            phase_labels = [data["train"]["meta"]["phase"][i] for i in indices]
            fig = plot2D_phase(walk_code, phase_labels, fig_title(f'content_by_{label}'))
        else:
            fig = plot2D(content_code_pca, data["train"]["meta"][label], fig_title(f'content_by_{label}'))
        add_fig(fig, f'content_by_{label}', "train")

    """
    fig = show_images_from_disk("", all_titles, 2, output_path + "all_codes")
    if summary:
        writers["train"].add_figure("all codes", fig, global_step=iter)
    """


def get_demo_plots(data, output_path):
    """
    data: {"train": dict_train, "test": dict_test}
    dict_train: {"style2d_code": blabla, etc.}
    """
    ensure_dirs(output_path)

    def fig_title(title):
        return pjoin(output_path, title)

    style_labels = data["train"]["meta"]["style"]

    adain_raw = []
    for key in ["style2d_adain", "style3d_adain"]:
        for phase in ["train", "test"]:
            adain_raw.append(data[phase][key])
    adain_tsne = calc_many_blas(adain_raw, calc_tsne)
    plot2D_overlay([adain_tsne[0], adain_tsne[2]],
                   [style_labels, style_labels],
                   [1.0, 0.5],
                   fig_title(f'joint_embedding_adain_tsne'))

    for key in ["style3d_code", "style3d_adain"]:
        tsne_code = calc_tsne(data["train"][key])
        plot2D(tsne_code, style_labels, fig_title(f'{key}_tsne'))

    content_code_pca = calc_pca(data["train"]["content_code"])

    indices = [i for i in range(len(data["train"]["meta"]["content"])) if data["train"]["meta"]["content"][i] == "walk"]
    walk_code = content_code_pca[np.array(indices)]
    phase_labels = [data["train"]["meta"]["phase"][i] for i in indices]
    plot2D_phase(walk_code, phase_labels, fig_title(f'content_by_phase'))

    plot2D(content_code_pca, style_labels, fig_title(f'content_by_style'))


def show_images_from_disk(path, titles, rows, this_title):
    images = []
    for title in titles:
        name = "%s.png" % title
        input_path = os.path.join(path, name)
        images.append(plt.imread(input_path))

    this_title = os.path.join(path, this_title)

    return show_images(images, titles, this_title, rows)


def show_images(images, titles, this_title, rows=1):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert (len(images) == len(titles))
    n_images = len(images)
    cols = np.ceil(n_images / float(rows))
    # if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    size = np.array((8, 8)) * np.array(rows, cols)
    fig = plt.figure(figsize=size)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        a.set_axis_off()
        plt.imshow(image)
        a.set_title(title)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)

    # plt.show()
    plt.savefig("%s.png" % this_title, dpi=150, bbox_inches='tight', pad_inches=0)

    return fig



