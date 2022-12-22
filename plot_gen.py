import argparse
import os
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
from utils.data_utils import ALPolicy
from minecore import pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and save plots on results')
    parser.add_argument('-p', '--policy', dest='policy', choices=('RS', 'US'), help='policy which generated results')
    parser.add_argument('-c', '--classifier', dest='classifier', choices=('Logistic Regression', 'SVM'),
                        default='SVM', help='classifier which generated the results')
    parser.add_argument('-a', '--aggregate', dest='aggregate', action='store_true', help='aggregate results across sizes')
    parser.add_argument('-t', '--cost-type', dest='cost_type', choices=('overall', 'annotation', 'misclass'), help='which cost to plot',
                        default='overall')
    parser.add_argument('--pairs', dest='pairs', nargs='+', default=[], help='pairs to plot')
    parser.add_argument('--phase', dest='phase', choices=('aut', 'man', 'resp', 'priv'), help='which phase costs to plot')
    parser.add_argument('--pair-average', dest='pair_average', action='store_true', help='if specified, average across all pairs')
    parser.add_argument('--load-path', dest='load_path', type=str, default='.data/active_learning/minecore',
                        help='path for loading data, defaults to .data/active_learning/minecore')
    parser.add_argument('--save-path', dest='save_path', type=str, default='',
                        help='path for saving plot, if left blank will show it and exit')

    args = parser.parse_args()
    if not args.aggregate:
        raise UserWarning('Not specifying -a/--aggregate has no effect at the moment')
    if args.policy == 'RS':
        policy = ALPolicy.RELEVANCE_SAMPLING
    else:
        policy = ALPolicy.UNCERTAINTY_SAMPLING

    result_files = os.listdir(args.load_path)
    dfs = []
    for file in result_files:
        split = file.split('_')
        if ALPolicy.from_string(split[0]) is not policy:
            continue
        pair = split[-1].split('.')[0]
        df = pd.read_csv(os.path.join(args.load_path, file), sep='\t', index_col=[0, 1]).groupby(level=1).mean()
        df = df[args.cost_type].filter(like=args.phase)
        dfs.append(df.rename(pair))
    df = pd.DataFrame(dfs[0]).join(dfs[1:])
    if args.pair_average:
        df = df.mean(axis=1)
        df.plot.bar(figsize=(12, 10))
        plt.xticks(rotation=30)
        plt.show()
        sys.exit(0)
    if args.pairs:
        plot_pairs = args.pairs
    else:
        plot_pairs = random.choices(df.columns.to_list(), k=5)
    df[plot_pairs].T.plot.bar(figsize=(12, 10))
    plt.title(f'{args.cost_type.capitalize()} MINECORE costs for {policy}')
    plt.xticks(rotation=30)
    if args.save_path:
        plt.savefig(args.save_path)
    plt.show()

