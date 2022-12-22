import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import itertools
import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm
from utils.misc import Colors, nested_dict_to_pandas
from utils.data_utils import ALPolicy, flatten, SLDConst
from utils.rcv1_class_bins import rcv1_class_bins, get_classes_by_prev, class_mapped_files, PREV_LOW, PREV_MEDIUM_LOW, PREV_MEDIUM_HIGH, PREV_HIGH, get_average_on_prev, get_all_prev_dfs
from sklearn.datasets import fetch_rcv1
from minecore import pairs

name_mapping = {
    '_priv_cost': '3rd phase',
    '_aut_cost': 'Fully automatic',
    '_man_cost': 'Fully manual',
    '_resp_cost': '2nd phase',
}


def rename_indices(idx):
    if isinstance(idx, str):
        replace = f"_{'_'.join(idx.split('_')[1:])}"
        return idx.replace(replace, f' {name_mapping[replace]}')
    return idx


def table_for_overall_column(dfs_by_policy, transpose=False, show_rand=True, show_sld=True):
    """
    This will create a table similar to the following for the 3rd Minecore phase. DFs must have a single column and must not have a multiindex.
    ---------------------------------------------------
                    Column Name (eg.Overall)
    ---------------------------------------------------
          ALvRS       Rand (RS)   ALvUS   Rand (US)   PL
    ---------------------------------------------------
    MLE     x               x       x           x     x
    SLD     x               x       x           x     x
    NP      x               x       x           x     x
    ---------------------------------------------------
    """
    data = {}
    for policy, df in dfs_by_policy.items():
        if policy is ALPolicy.UNCERTAINTY_SAMPLING or policy is ALPolicy.RELEVANCE_SAMPLING or policy is ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING:
            if show_rand:
                data[f'Rand {policy.compact_str()}'] = {'Pre-SLD': df.loc['Rand_priv_cost'].item(),
                                          'Post-SLD': df.loc['Rand SLD_priv_cost'].item()}
            else:
                data[str(policy)] = {'Pre-SLD': df.loc['AL_priv_cost'].item(), 'Post-SLD': df.loc['AL SLD_priv_cost'].item()}
        else:
            if not show_rand:
                data[str(policy)] = {'Pre-SLD': df.loc['MLE_priv_cost'].item(), 'Post-SLD': df.loc['SLD_priv_cost'].item()}
    df = pd.DataFrame(data)
    if not show_rand:
        df = df[sorted(df.columns, key=ALPolicy.from_string)]
    if not show_sld:
        df.drop(index=['Post-SLD'], inplace=True)
    if transpose:
        df = df.T
    print(df.to_latex(float_format="%.2f"))


def table_with_multiple_cs(cs_dfs):
    def bold_best(row):
        l = [""] * len(row)
        s = row.argsort()
        l[s[0]] = 'textbf:--rwrap'
        return l
    if not args.show_rand:
        data = [{str(k): {'Pre-SLD': v['AL_priv_cost'] if 'AL_priv_cost' in v else v['MLE_priv_cost'],
                          'Post-SLD': v['AL SLD_priv_cost'] if 'AL SLD_priv_cost' in v else v['SLD_priv_cost']} for k, v in
                 map(lambda a: (a[0], a[1].to_dict()['overall']), d.items())} for d in cs_dfs]
    else:
        data = [{str(k.to_rand()): {'Pre-SLD': v['Rand_priv_cost'],
                          'Post-SLD': v['Rand SLD_priv_cost']} for k, v in
                 filter(lambda a: 'Rand_priv_cost' in a[1], map(lambda a: (a[0], a[1].to_dict()['overall']),
                                                                d.items()))} for d in cs_dfs]

    ddf = {p[0][0]: [t[1] for t in p] for p in zip(*map(dict.items, data))}
    ddf = {k: {ii[0][0]: [t[1] for t in ii] for ii in zip(*map(dict.items, v))} for k, v in ddf.items()}
    df = nested_dict_to_pandas(ddf)
    df = df.melt()
    df['CS'] = np.tile([1,2,3], len(df) // 3)
    df = df.pivot(index=['CS', 'variable_1'], columns='variable_0', values='value')
    df = df.loc[[(i+1, s) for i in range(len(df) // 2) for s in ('Pre-SLD', 'Post-SLD')]]
    df = df[sorted(df.columns, key=ALPolicy.from_string)]
    df.columns.name = '$\Lambda$'
    df.index.names = ['', '']
    if not args.show_sld:
        df = df.drop('Post-SLD', level=1).droplevel(1)
    df = df.rename({1: '$\Lambda_1$', 2: '$\Lambda_2$', 3: '$\Lambda_3$'})
    if args.show_sld:
        df = df.rename(columns=lambda c: '\multicolumn{1}{c}{%s}' % c)
        print(df.style.format('{:.2f}').apply(bold_best, axis=1).to_latex(column_format='lr|rrrr', clines='all;data'))
    else:
        print(df.style.format('{:.2f}').apply(bold_best, axis=1).to_latex(column_format='lrrrr', hrules=True))


def create_bins():
    rcv1 = fetch_rcv1()
    resp_classes = set(map(lambda p: p[0], pairs))
    priv_classes = set(map(lambda p: p[1], pairs))
    y = rcv1.target.toarray().squeeze()
    labels = rcv1.target_names.tolist()
    resp_y = y[:, [labels.index(r) for r in resp_classes]]
    priv_y = y[:, [labels.index(p) for p in priv_classes]]
    resp_binned = rcv1_class_bins(resp_classes, resp_y)
    priv_binned = rcv1_class_bins(priv_classes, priv_y)
    resp_low_prev, resp_mid_low_rev, resp_mid_high_prev, resp_high_prev = get_classes_by_prev(resp_binned, PREV_LOW), \
                                                                          get_classes_by_prev(resp_binned,
                                                                                              PREV_MEDIUM_LOW), \
                                                                          get_classes_by_prev(resp_binned,
                                                                                              PREV_MEDIUM_HIGH), \
                                                                          get_classes_by_prev(resp_binned, PREV_HIGH)
    priv_low_prev, priv_mid_low_rev, priv_mid_high_prev, priv_high_prev = get_classes_by_prev(priv_binned, PREV_LOW), \
                                                                          get_classes_by_prev(priv_binned,
                                                                                              PREV_MEDIUM_LOW), \
                                                                          get_classes_by_prev(priv_binned,
                                                                                              PREV_MEDIUM_HIGH), \
                                                                          get_classes_by_prev(priv_binned, PREV_HIGH)
    return resp_low_prev, resp_mid_low_rev, resp_mid_high_prev, resp_high_prev, priv_low_prev, priv_mid_low_rev, \
           priv_mid_high_prev, priv_high_prev


def __collect_data_for_wilcoxon(files):
    pairs_data = {}
    used_policies = list(files.keys())
    for f in tqdm(files[used_policies[0]]):
        fs_pd = {used_policies[0]: pd.read_csv(f, sep='\t', index_col=[0, 1])}
        for pol in used_policies[1:]:
            if pol is ALPolicy.PASSIVE_LEARNING:
                pl_root = os.path.split(files[pol][0])[0]
                pl_path = os.path.join(pl_root, os.path.split(f)[-1].replace(str(used_policies[0]),
                                                                             str(pol)))
                fs_pd[pol] = pd.read_csv(pl_path, sep='\t', index_col=[0, 1])
            else:
                fs_pd[pol] = pd.read_csv(f.replace(str(used_policies[0]), str(pol)), sep='\t', index_col=[0, 1])

        # aggregate by size
        fs_pd = {k: v.groupby(level=1).mean() for k, v in fs_pd.items()}
        if args.binning:
            if args.show_rand:
                try:
                    for pol in used_policies:
                        p = pairs_data.setdefault(pol.to_rand(), [])
                        p.append(fs_pd[pol]['overall']['Rand_priv_cost'])
                        ps = pairs_data.setdefault(SLDConst.from_al_policy(pol.to_rand()), [])
                        ps.append(fs_pd[pol]['overall']['Rand SLD_priv_cost'])
                except ValueError:
                    pass
            else:
                for pol in used_policies:
                    p = pairs_data.setdefault(pol, [])
                    if v := fs_pd[pol]['overall'].get('AL_priv_cost'):
                        p.append(v)
                    else:
                        p.append(fs_pd[pol]['overall']['MLE_priv_cost'])
                    ps = pairs_data.setdefault(SLDConst.from_al_policy(pol), [])
                    if v := fs_pd[pol]['overall'].get('AL SLD_priv_cost'):
                        ps.append(v)
                    else:
                        ps.append(fs_pd[pol]['overall']['SLD_priv_cost'])
        elif args.show_rand:
            try:
                for pol in used_policies:
                    if not args.show_sld:
                        p = pairs_data.setdefault(pol.to_rand(), [])
                        p.append(fs_pd[pol]['overall']['Rand_priv_cost'])
                    else:
                        ps = pairs_data.setdefault(SLDConst.from_al_policy(pol.to_rand()), [])
                        ps.append(fs_pd[pol]['overall']['Rand SLD_priv_cost'])
            except ValueError:
                pass
        elif args.show_sld:
            for pol in used_policies:
                p = pairs_data.setdefault(pol, [])
                if v := fs_pd[pol]['overall'].get('AL SLD_priv_cost'):
                    p.append(v)
                else:
                    p.append(fs_pd[pol]['overall']['SLD_priv_cost'])
        else:
            for pol in used_policies:
                p = pairs_data.setdefault(pol, [])
                if v := fs_pd[pol]['overall'].get('AL_priv_cost'):
                    p.append(v)
                else:
                    p.append(fs_pd[pol]['overall']['MLE_priv_cost'])
    return pairs_data


def wilcoxon_test(res_files, cs):
    if args.binning:
        names = ['Resp Low', 'Resp Mid-Low', 'Resp Mid-High', 'Resp High', 'Priv Low', 'Priv Mid-Low', 'Priv Mid-High',
                 'Priv High']
        bins = dict(zip(names, create_bins()))
        cls_map_resp, cls_map_priv = class_mapped_files(res_files[ALPolicy.RELEVANCE_SAMPLING])
        pl_cls_map_resp, pl_cls_map_priv = class_mapped_files(res_files[ALPolicy.PASSIVE_LEARNING])
        for name, clss in bins.items():
            if 'Resp' in name:
                pairs_files = list(flatten(map(cls_map_resp.get, clss)))
                pl_pair_files = list(flatten(map(pl_cls_map_resp.get, clss)))
            else:
                pairs_files = list(flatten(map(cls_map_priv.get, clss)))
                pl_pair_files = list(flatten(map(pl_cls_map_priv.get, clss)))

            # dict for compatibility with the else case
            pairs_data = __collect_data_for_wilcoxon({ALPolicy.RELEVANCE_SAMPLING: pairs_files,
                                                      ALPolicy.UNCERTAINTY_SAMPLING: [],
                                                      ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING: [],
                                                      ALPolicy.PASSIVE_LEARNING: pl_pair_files})
            it = itertools.combinations(pairs_data.keys(), 2)
            for pol1, pol2 in filter(lambda pols: (isinstance(pols[0], ALPolicy) and pols[0].is_self_with_sld(pols[1]))
                                                  or (isinstance(pols[1], ALPolicy) and pols[1].is_self_with_sld(pols[0])), it):
                print(f'Comparing {pol1} with {pol2}, bin {name}; SLD: {args.show_sld}, Rand: {args.show_rand}. '
                      f'Cost structure: {cs}')
                print(wilcoxon(pairs_data[pol1], pairs_data[pol2]))
    else:
        pairs_data = __collect_data_for_wilcoxon(res_files)
        it = itertools.combinations(pairs_data.keys(), 2)
        for pol1, pol2 in it:
            print(f'Comparing {pol1} with {pol2}; SLD: {args.show_sld}, Rand: {args.show_rand}. '
                  f'Cost structure: {cs}')
            print(wilcoxon(pairs_data[pol1], pairs_data[pol2]))


def elaborate_on_cost_structure(cs):
    res_files = {}
    for res_fold in args.results:
        results = os.listdir(res_fold)
        results.sort()
        results = filter(lambda r: int(r.split('_')[-1].split('.')[0][2:]) == cs and os.path.splitext(r)[-1] == '.tsv', results)
        for policy, group in itertools.groupby(results, key=lambda k: k.split('_')[0]):
            res_files[ALPolicy.from_string(policy)] = list(map(lambda r: os.path.join(res_fold, r), group))

    if args.wilcoxon:
        wilcoxon_test(res_files, cs)
        return None, None

    assert args.aggregate_on_pairs ^ (args.pairs_to_show is not None) ^ args.binning, \
        'Only one of aggregate-on-pairs, pairs-to-show or binning must be true'

    if args.binning:
        resp_low_prev, resp_mid_low_rev, resp_mid_high_prev, resp_high_prev, priv_low_prev, priv_mid_low_rev, \
        priv_mid_high_prev, priv_high_prev = create_bins()

    dfs_by_policy = {}
    resp_priv_prev_dfs = [{}, {}]
    for policy, files in tqdm(res_files.items()):
        if args.aggregate_on_pairs:
            df = sum(pd.read_csv(f, sep='\t', index_col=[0, 1]) for f in files) / len(files)
            if args.aggregate_on_size:
                df = df.groupby(level=1).mean()
                dfs_by_policy[policy] = df.reindex(args.rows_to_show).dropna()[args.columns_to_show]
            else:
                sizes = set(map(lambda i: i[0], df.index))
                rows = [(s, r) for r in set(i[1] for i in df.index) for s in sizes]
                rows.sort()
                dfs_by_policy[policy] = df.reindex(rows).dropna()[args.columns_to_show]
        elif args.binning:
            cls_map_resp, cls_map_priv = class_mapped_files(files)
            # Responsive
            if 'Rand_priv_cost' in args.rows_to_show and policy is ALPolicy.PASSIVE_LEARNING:
                continue
            rows_to_show = args.pl_rows_to_show if policy is ALPolicy.PASSIVE_LEARNING else args.rows_to_show
            if 'Rand_priv_cost' in rows_to_show:
                policy = policy.to_rand()

            low_prev_df_resp, med_low_prev_df_resp, med_high_prev_df_resp, high_prev_df_resp = list(get_all_prev_dfs(cls_map_resp, [resp_low_prev, resp_mid_low_rev, resp_mid_high_prev, resp_high_prev], rows_to_show, args.columns_to_show))
            # Privilege
            low_prev_df_priv, med_low_prev_df_priv, med_high_prev_df_priv, high_prev_df_priv = list(get_all_prev_dfs(cls_map_priv, [priv_low_prev, priv_mid_low_rev, priv_mid_high_prev, priv_high_prev], rows_to_show, args.columns_to_show))

            resp_priv_prev_dfs[0][str(policy)] = {PREV_LOW: low_prev_df_resp, PREV_MEDIUM_LOW: med_low_prev_df_resp,
                                                  PREV_MEDIUM_HIGH: med_high_prev_df_resp, PREV_HIGH: high_prev_df_resp}
            resp_priv_prev_dfs[1][str(policy)] = {PREV_LOW: low_prev_df_priv, PREV_MEDIUM_LOW: med_low_prev_df_priv,
                                                  PREV_MEDIUM_HIGH: med_high_prev_df_priv, PREV_HIGH: high_prev_df_priv}
        else:
            # show_pairs(args.pairs_to_show)
            pass
    return dfs_by_policy, resp_priv_prev_dfs


def plot_over_sizes(dfs):
    for c, df in enumerate(dfs):
        policies = list(df.keys())
        sizes = set([i[0] for i in df[policies[0]].index]) - {1000}
        data = {}
        for size in sizes:
            # The 23k correction is here due to a former bug where we used to save tr sets with size 23149 with label 23000
            data[size] = {str(p): d.loc[(23149 if p is ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING and size == 23000 else size,
                                    'AL_priv_cost' if p is not ALPolicy.PASSIVE_LEARNING else 'MLE_priv_cost')].item()
                          for p, d in df.items()}
        data[23149] = data[23000]
        del data[23000]
        pl = pd.DataFrame(data).T.sort_index()
        pl.plot(title=f'Cost structure {c+1}', xlabel='Tr. size $|\mathcal{L}|$', ylabel='Cost', yticks=[])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    c = Colors()
    parser = argparse.ArgumentParser(f'Elaborate MINECORE results (see {c.make_bold("run_minecore.py")})')
    parser.add_argument('-r', '--results', nargs='+', help='path(s) to results folder(s)')
    parser.add_argument('--aggregate-on-size', action='store_true', help='aggregate results on size')
    parser.add_argument('--aggregate-on-pairs', action='store_true', help='aggregate results on pairs')
    parser.add_argument('--binning', action='store_true', help='aggregate in bins with low, medium-low, medium-high and high prev.')
    parser.add_argument('--show-rand', action='store_true', help='show Rand policy results')
    parser.add_argument('--show-sld', action='store_true', help='whether to show or not results from SLD')
    parser.add_argument('--pairs-to-show', nargs='*', help='pairs to show if not aggregating by pairs')
    parser.add_argument('--columns-to-show', nargs='*', help='columns to show', default=['overall'])
    parser.add_argument('--rows-to-show', nargs='*', help='rows to show')
    parser.add_argument('--pl-rows-to-show', nargs='*', help='rows to show for PL policy')
    parser.add_argument('--cost-structure', dest='cost_structure', nargs='+', type=int, choices=(1, 2, 3), help='Use cost structure 1, 2 or 3', default=1)
    parser.add_argument('-w', '--wilcoxon', action='store_true', help='perform wilcoxon test')
    parser.add_argument('--plot-over-sizes', action='store_true')
    args = parser.parse_args()

    cs_dfs, cs_resp_priv_prevs = [], []
    for cs in args.cost_structure:
        dfs_by_policy, resp_priv_prev_dfs = elaborate_on_cost_structure(cs)
        cs_dfs.append(dfs_by_policy)
        cs_resp_priv_prevs.append(resp_priv_prev_dfs)

    if args.plot_over_sizes:
        plot_over_sizes(cs_dfs)
        exit()
    # for policy, df in dfs_by_policy.items():
    #     print(f'Policy {policy}')
    #     print(df.rename(rename_indices).rename(columns=lambda a: a.capitalize()))
    if not args.wilcoxon and len(args.columns_to_show) == 1:
        if not args.binning:
            #table_for_overall_column(dfs_by_policy, transpose=False, show_rand=args.show_rand, show_sld=args.show_sld)
            table_with_multiple_cs(cs_dfs)
        else:
            # This mess down here is to create a MultiIndex similar to the one in the paper
            for cs, resp_priv_prev_dfs in enumerate(cs_resp_priv_prevs):
                policies = list(resp_priv_prev_dfs[1].keys())
                policies.sort(key=lambda k: ALPolicy.from_string(k))
                resp, priv = resp_priv_prev_dfs
                resp = pd.DataFrame({k: {ki: dict(vi[args.columns_to_show[0]]) for ki, vi in v.items()} for k, v in resp.items()})
                priv = pd.DataFrame({k: {ki: dict(vi[args.columns_to_show[0]]) for ki, vi in v.items()} for k, v in priv.items()})
                resp = resp.stack().apply(pd.Series).unstack().swaplevel(0, 1, axis=1).sort_index(axis=1, ascending=False)[policies]
                priv = priv.stack().apply(pd.Series).unstack().swaplevel(0, 1, axis=1).sort_index(axis=1, ascending=False)[policies]
                phases = ['Pre-SLD', 'Post-SLD']
                for policy in policies:
                    r = ((resp[[(policy, phases[1])]].to_numpy() - resp[[(policy, phases[0])]].to_numpy()) / resp[[(policy, phases[1])]].to_numpy()) * 100
                    resp[(policy, 'Increase')] = np.array(list(map(lambda i: [f'+{i[0]:.3f}%' if i[0] > 0 else f'{i[0]:.3f}%'], r)))
                    p = ((priv[[(policy, phases[1])]].to_numpy() - priv[[(policy, phases[0])]].to_numpy()) / priv[
                        [(policy, phases[1])]].to_numpy()) * 100
                    priv[(policy, 'Increase')] = np.array(list(map(lambda i: [f'+{i[0]:.3f}%' if i[0] > 0 else f'{i[0]:.3f}%'], p)))
                phases.append('Increase')
                resp = resp[list(itertools.product(policies, phases))]
                priv = priv[list(itertools.product(policies, phases))]
                print(f'{"#" * 10} CS {cs} {"#" * 10}')
                print(f'{"#" * 10} RESPONSIVE {"#" * 10}')
                print(resp.to_latex(float_format='%.2f'))
                print("#" * 20)
                print(f'{"#" * 10} PRIVILEGED {"#" * 10}')
                print(priv.to_latex(float_format='%.2f'))
                input()

