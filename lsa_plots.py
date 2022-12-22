from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_rcv1
from utils.data_utils import ALPolicy, flatten, random_sample_from_dataset, get_xy_with_given_pos_and_negs
from minecore import pairs
from tqdm import tqdm
import copy
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pickle
import glob
import argparse
import random


def do_plot(policy, cls, y_c, x_svd, idx, initial_idxs):
    fig, ax = plt.subplots()
    mask = np.zeros(100_000, dtype=int)
    mask[idx] = 1
    mask[np.logical_and(mask == 1, y_c == 1)] = 2
    if initial_idxs is not None:
        mask[initial_idxs] = 0
    marker = 'o'
    ax.scatter(x_svd[:, 0][y_c == 0], x_svd[:, 1][y_c == 0], alpha=0.3, label='negatives', marker=marker)
    ax.scatter(x_svd[:, 0][y_c == 1], x_svd[:, 1][y_c == 1], alpha=0.4, c='orange', label='positives', marker=marker)
    ax.scatter(x_svd[:, 0][mask == 1], x_svd[:, 1][mask == 1], alpha=0.5, facecolors='none', edgecolors='r',
               label='selected (neg)', marker=marker)
    ax.scatter(x_svd[:, 0][mask == 2], x_svd[:, 1][mask == 2], alpha=0.5, facecolors='none', edgecolors='g',
               label='selected (pos)', marker=marker)
    if args.show_zoom:
        axin = ax.inset_axes([0.2, 0.1, 0.3, 0.3], transform=ax.transData)
        min_area_x, min_area_y = x_svd[mask > 0].min(0)
        max_area_x, max_area_y = x_svd[mask > 0].max(0)
        marker_size = mpl.rcParams['lines.markersize'] ** 1.5
        axin.scatter(x_svd[:, 0][y_c == 0], x_svd[:, 1][y_c == 0], alpha=0.3, c='#1F77B4', s=marker_size, marker=marker)
        axin.scatter(x_svd[:, 0][y_c == 1], x_svd[:, 1][y_c == 1], alpha=0.4, c='orange', label='positives', s=marker_size, marker=marker)
        axin.scatter(x_svd[:, 0][mask == 1], x_svd[:, 1][mask == 1], alpha=0.5, facecolors='none', edgecolors='r', s=marker_size, marker=marker)
        axin.scatter(x_svd[:, 0][mask == 2], x_svd[:, 1][mask == 2], alpha=0.5, facecolors='none', edgecolors='g', s=marker_size, marker=marker)
        axin.set_xlim(min_area_x, max_area_x)
        axin.set_ylim(min_area_y, max_area_y)
        axin.set_xticklabels([])
        axin.set_yticklabels([])
        ax.indicate_inset_zoom(axin, edgecolor='black')
    fig.suptitle(f'{policy}. # Annotations: {(len(idx) - 1000) or 1000}; RCV1-v2 class: {cls}')
    plt.legend(loc='lower left')
    fig.tight_layout()
    if args.save_path:
        fig.savefig(os.path.join(args.save_path, f'lsa_plot_{policy}_{cls}_ann-{len(idx)}.png'))
    else:
        plt.show()
    plt.close()


def lsa_plot(policy: ALPolicy, cls, y_c, x_svd):
    files = sorted(glob.glob(os.path.join(args.load_path, cls, f'*{policy}*pool1*')), key=lambda f: int(f.split('checkpoint-')[-1].split('_')[0]))
    initial_idxs = []
    for f in files:
        with open(f, 'rb') as o:
            idx = pickle.load(o)
            if len(idx) == 2:
                idx = idx[0]
            size = int(f.split('checkpoint-')[-1].split('_')[0])
            if size == 1000:
                initial_idxs = idx
                y_c[initial_idxs] = -1
                continue
            if size not in args.sizes:
                continue
        if args.plot_rand:
            y_c_idx = y_c[idx]
            pos = (y_c_idx == 1).sum()
            neg = (y_c_idx == 0).sum()
            _, _, idx = get_xy_with_given_pos_and_negs(x_svd, y_c, pos, neg, seed=42)
            policy = policy.to_rand()
        do_plot(copy.copy(policy), cls, y_c, x_svd, idx, initial_idxs)


def parallel_plot(cls, y_c, policies, x_svd):
    for p in policies:
        policy = ALPolicy.from_string(p)
        if policy is ALPolicy.PASSIVE_LEARNING:
            idxs = np.random.choice(np.arange(len(x_svd)), replace=False, size=1000)
            do_plot(policy, cls, y_c, x_svd, idxs, None)
        else:
            lsa_plot(policy, cls, y_c, x_svd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Scatter plot for Active Learning')
    parser.add_argument('-c', '--rclass', nargs='+', help='RCV1 class')
    parser.add_argument('-r', '--randclass', type=int, help='use n random classes. -c must not be specified')
    parser.add_argument('-j', '--jobs', type=int, help='if -1 use one process for each class')
    parser.add_argument('--load-path', help='loading path', default='.data/active_learning')
    parser.add_argument('--save-path', help='saving path for plot')
    parser.add_argument('-p', '--policy', choices=['RS', 'US', 'PL', 'RUS'], nargs='+')
    parser.add_argument('-s', '--sizes', type=int, nargs='+')
    parser.add_argument('--show-zoom', action='store_true')
    parser.add_argument('--plot-rand', action='store_true')
    args = parser.parse_args()

    assert (args.rclass is not None) ^ (args.randclass is not None), 'only one of -c or -r can be specified'

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
    rcv1 = fetch_rcv1()
    x = rcv1.data[:100_000]
    x_svd = TruncatedSVD().fit_transform(x)
    possible_classes = list(set(flatten(pairs)))
    y = rcv1.target[:100_000].toarray().squeeze()
    r_classes = random.sample(possible_classes, k=args.randclass) if args.randclass else args.rclass

    jobs = len(r_classes) if args.jobs == -1 else args.jobs
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        futures = []
        for cls in r_classes:
            cls_idx = rcv1.target_names.tolist().index(cls)
            futures.append(pool.submit(parallel_plot, cls, y[:, cls_idx], args.policy, x_svd))

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            f.result()
