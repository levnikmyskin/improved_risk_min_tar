import argparse
import copy
import os
import pandas as pd
import pickle
import concurrent.futures
import warnings
from utils.data_utils import ALPolicy
from minecore import MineCore, cost_structure_3, cost_structure_2, cost_structure_1, Costs, pairs
from utils import filter_file, aggregate_same_sizes
from tqdm import tqdm


def __get_data_from_result_file(file, passive_learning=False):
    with open(file, 'rb') as f:
        results = pickle.load(f)
    if passive_learning:
        y_te = results['y_te']
        np_y_te = results['np_y_te']
        mle_probs = results['mle_probs']
        sld_probs = results['sld_post']
        np_probs = results['np_probs']
        return y_te, np_y_te, mle_probs, sld_probs, np_probs
    al_y_te = results['al_y_te']
    rand_y_te = results['rand_y_te']
    np_y_te = results['np_y_te']
    al_probs = results['al_probs']
    al_sld_probs = results['al_sld_posteriors']
    rand_probs = results['rand_probs']
    rand_sld_probs = results['rand_sld_posteriors']
    al_np_probs = results['al_np_probs']
    rand_np_probs = results['rand_np_probs']
    return al_y_te, rand_y_te, np_y_te, al_probs, al_sld_probs, rand_probs, rand_sld_probs, al_np_probs, rand_np_probs


def __get_costs_for_run(costs: Costs, tau_rs, tau_ps, cm_2, cm_3, posteriors, n_docs, prefix):
    costs.posterior_probabilities = posteriors
    map_fn = lambda r: {'overall': list(r[0].values())[0], 'annotation': list(r[1].values())[0], 'misclass': list(r[2].values())[0]}
    return {
        f'{prefix}_man_cost': map_fn(costs.get_manual_costs(n_docs)),
        f'{prefix}_aut_cost': map_fn(costs.get_automatic_costs()),
        f'{prefix}_resp_cost': map_fn(costs.get_second_phase_costs(cm_2, tau_rs)),
        f'{prefix}_priv_cost': map_fn(costs.get_third_phase_costs(cm_3, tau_rs, tau_ps))
    }


def run_minecore_on_file(cr_file, cp_file, pair, cost_structure):
    c_r, c_p = pair
    cr_al_y_te, cr_rand_y_te, cr_np_y_te, cr_al_probs, cr_al_sld_probs, cr_rand_probs, cr_rand_sld_probs, cr_al_np_probs, cr_rand_np_probs = __get_data_from_result_file(cr_file)
    cp_al_y_te, cp_rand_y_te, cp_np_y_te, cp_al_probs, cp_al_sld_probs, cp_rand_probs, cp_rand_sld_probs, cp_al_np_probs, cp_rand_np_probs = __get_data_from_result_file(cp_file)
    al_costs = Costs(cost_structure, [(c_r, c_p)], None, y_arr={c_r: cr_al_y_te, c_p: cp_al_y_te})
    al_sld_costs = copy.deepcopy(al_costs)
    rand_costs = Costs(cost_structure, [(c_r, c_p)], None, y_arr={c_r: cr_rand_y_te, c_p: cp_rand_y_te})
    rand_sld_costs = copy.deepcopy(rand_costs)
    al_np_costs = Costs(cost_structure, [(c_r, c_p)], None, y_arr={c_r: cr_np_y_te, c_p: cp_np_y_te})
    rand_np_costs = copy.deepcopy(al_np_costs)

    al_minecore = MineCore(None, None, {c_r: cr_al_probs, c_p: cp_al_probs}, {c_r: cr_al_y_te, c_p: cp_al_y_te}, None, 1., 1.)
    al_sld_minecore = MineCore(None, None, {c_r: cr_al_sld_probs, c_p: cp_al_sld_probs},
                               {c_r: cr_al_y_te, c_p: cp_al_y_te}, None, 1., 1.)
    rand_minecore = MineCore(None, None, {c_r: cr_rand_probs, c_p: cp_rand_probs},
                             {c_r: cr_rand_y_te, c_p: cp_rand_y_te}, None, 1., 1.)
    rand_sld_minecore = MineCore(None, None, {c_r: cr_rand_sld_probs, c_p: cp_rand_sld_probs},
                                 {c_r: cr_rand_y_te, c_p: cp_rand_y_te}, None, 1., 1.)
    al_np_minecore = MineCore(None, None, {c_r: cr_al_np_probs, c_p: cp_al_np_probs},
                              {c_r: cr_np_y_te, c_p: cp_np_y_te}, None, 1., 1.)
    rand_np_minecore = MineCore(None, None, {c_r: cr_rand_np_probs, c_p: cp_rand_np_probs},
                                {c_r: cr_np_y_te, c_p: cp_np_y_te}, None, 1., 1.)

    al_cm_2, al_cm_3 = al_minecore.run_on_pair(c_r, c_p, al_costs)
    al_tau_rs, al_tau_ps = al_minecore.tau_rs, al_minecore.tau_ps
    al_sld_cm_2, al_sld_cm_3 = al_sld_minecore.run_on_pair(c_r, c_p, al_sld_costs)
    al_sld_tau_rs, al_sld_tau_ps = al_sld_minecore.tau_rs, al_sld_minecore.tau_ps
    rand_cm_2, rand_cm_3 = rand_minecore.run_on_pair(c_r, c_p, rand_costs)
    rand_tau_rs, rand_tau_ps = rand_minecore.tau_rs, rand_minecore.tau_ps
    rand_sld_cm_2, rand_sld_cm_3 = rand_sld_minecore.run_on_pair(c_r, c_p, rand_sld_costs)
    rand_sld_tau_rs, rand_sld_tau_ps = rand_sld_minecore.tau_rs, rand_sld_minecore.tau_ps
    al_np_cm_2, al_np_cm_3 = al_np_minecore.run_on_pair(c_r, c_p, al_np_costs)
    al_np_tau_rs, al_np_tau_ps = al_np_minecore.tau_rs, al_np_minecore.tau_ps
    rand_np_cm_2, rand_np_cm_3 = rand_np_minecore.run_on_pair(c_r, c_p, rand_np_costs)
    rand_np_tau_rs, rand_np_tau_ps = rand_np_minecore.tau_rs, rand_np_minecore.tau_ps

    if args.save_contingency:
        with open(os.path.join(args.save_path, f'{policy}_minecorecontingencies_{"-".join(pair)}_cs{args.cost_structure}.pkl'), 'wb') as f:
            pickle.dump({f'{str(policy)} pre-SLD': al_cm_3,
                         f'{str(policy)} post-SLD': al_sld_cm_3,
                         f'Rand ({policy.compact_str()}) pre-SLD': rand_cm_3,
                         f'Rand ({policy.compact_str()}) post-SLD': rand_sld_cm_3}, f)
        return None

    al_mc_costs = __get_costs_for_run(al_costs, al_tau_rs, al_tau_ps, al_cm_2, al_cm_3, {c_r: cr_al_probs, c_p: cp_al_probs}, len(cr_al_y_te), 'AL')
    al_mc_sld_costs = __get_costs_for_run(al_sld_costs, al_sld_tau_rs, al_sld_tau_ps, al_sld_cm_2, al_sld_cm_3, {c_r: cr_al_sld_probs, c_p: cp_al_sld_probs}, len(cr_al_y_te), 'AL SLD')
    rand_mc_costs = __get_costs_for_run(rand_costs, rand_tau_rs, rand_tau_ps, rand_cm_2, rand_cm_3, {c_r: cr_rand_probs, c_p: cp_rand_probs}, len(cr_rand_y_te), 'Rand')
    rand_mc_sld_costs = __get_costs_for_run(rand_sld_costs, rand_sld_tau_rs, rand_sld_tau_ps, rand_sld_cm_2, rand_sld_cm_3, {c_r: cr_rand_sld_probs, c_p: cp_rand_sld_probs}, len(cr_rand_y_te), 'Rand SLD')
    al_mc_np_costs = __get_costs_for_run(al_np_costs, al_np_tau_rs, al_np_tau_ps, al_np_cm_2, al_np_cm_3, {c_r: cr_al_np_probs, c_p: cp_al_np_probs}, len(cr_np_y_te), 'AL NP')
    rand_mc_np_costs = __get_costs_for_run(rand_np_costs, rand_np_tau_rs, rand_np_tau_ps, rand_np_cm_2, rand_np_cm_3, {c_r: cr_rand_np_probs, c_p: cp_rand_np_probs}, len(cr_np_y_te), 'Rand NP')

    al_mc_costs.update(al_mc_sld_costs)
    al_mc_costs.update(rand_mc_costs)
    al_mc_costs.update(rand_mc_sld_costs)
    al_mc_costs.update(al_mc_np_costs)
    al_mc_costs.update(rand_mc_np_costs)
    return al_mc_costs


def run_minecore_on_passive_learning_file(cr_file, cp_file, pair, cost_structure):
    c_r, c_p = pair
    cr_y_te, cr_np_y_te, cr_mle_probs, cr_sld_probs, cr_np_probs = __get_data_from_result_file(cr_file, passive_learning=True)
    cp_y_te, cp_np_y_te, cp_mle_probs, cp_sld_probs, cp_np_probs = __get_data_from_result_file(cp_file, passive_learning=True)
    mle_costs = Costs(cost_structure, [(c_r, c_p)], None, y_arr={c_r: cr_y_te, c_p: cp_y_te})
    sld_costs = copy.deepcopy(mle_costs)
    np_costs = Costs(cost_structure, [(c_r, c_p)], None, y_arr={c_r: cr_np_y_te, c_p: cp_np_y_te})

    mle_minecore = MineCore(None, None, {c_r: cr_mle_probs, c_p: cp_mle_probs}, {c_r: cr_y_te, c_p: cp_y_te}, None, 1., 1.)
    sld_minecore = MineCore(None, None, {c_r: cr_sld_probs, c_p: cp_sld_probs}, {c_r: cr_y_te, c_p: cp_y_te}, None, 1., 1.)
    np_minecore = MineCore(None, None, {c_r: cr_np_probs, c_p: cp_np_probs}, {c_r: cr_np_y_te, c_p: cp_np_y_te}, None, 1., 1.)

    mle_cm_2, mle_cm_3 = mle_minecore.run_on_pair(c_r, c_p, mle_costs)
    mle_tau_rs, mle_tau_ps = mle_minecore.tau_rs, mle_minecore.tau_ps
    sld_cm_2, sld_cm_3 = sld_minecore.run_on_pair(c_r, c_p, sld_costs)
    sld_tau_rs, sld_tau_ps = sld_minecore.tau_rs, sld_minecore.tau_ps
    np_cm_2, np_cm_3 = np_minecore.run_on_pair(c_r, c_p, np_costs)
    np_tau_rs, np_tau_ps = np_minecore.tau_rs, np_minecore.tau_ps

    if args.save_contingency:
        with open(os.path.join(args.save_path, f'{policy}_minecorecontingencies_{"-".join(pair)}_cs{args.cost_structure}.pkl'), 'wb') as f:
            pickle.dump({f'{str(policy)} pre-SLD': mle_cm_3,
                         f'{str(policy)} post-SLD': sld_cm_3}, f)
        return None

    mle_mc_costs = __get_costs_for_run(mle_costs, mle_tau_rs, mle_tau_ps, mle_cm_2, mle_cm_3, {c_r: cr_mle_probs, c_p: cp_mle_probs}, len(cr_y_te), 'MLE')
    sld_mc_costs = __get_costs_for_run(sld_costs, sld_tau_rs, sld_tau_ps, sld_cm_2, sld_cm_3, {c_r: cr_sld_probs, c_p: cp_sld_probs}, len(cr_y_te), 'SLD')
    np_mc_costs = __get_costs_for_run(np_costs, np_tau_rs, np_tau_ps, np_cm_2, np_cm_3, {c_r: cr_np_probs, c_p: cp_np_probs}, len(cr_np_y_te), 'NP')

    mle_mc_costs.update(sld_mc_costs)
    mle_mc_costs.update(np_mc_costs)
    return mle_mc_costs


def minecore_costs_on_pair(pair, label_files, passive_learning=False):
    c_r, c_p = pair
    if c_r not in label_files or c_p not in label_files:
        return f"Either {c_r} or {c_p} was not available.", None
    c_r_files = label_files[c_r]
    c_p_files = label_files[c_p]
    sizes = sorted(c_r_files.keys() & c_p_files.keys())
    results = {}
    for size in sizes:
        c_r_file, c_p_file = c_r_files[size], c_p_files[size]
        if passive_learning:
            results[size] = run_minecore_on_passive_learning_file(c_r_file, c_p_file, (c_r, c_p), cost_structure)
        else:
            results[size] = run_minecore_on_file(c_r_file, c_p_file, (c_r, c_p), cost_structure)
    return results, pair


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MINECORE with probs. obtained from either ALvRS or ALvUS and "
                                                 "random dataset")
    parser.add_argument('-p', '--policy', dest='policy', choices=('RS', 'US', 'PL', 'RUS'), help='policy which generated results')
    parser.add_argument('-c', '--classifier', dest='classifier', choices=('Logistic Regression', 'SVM'),
                        default='Logistic Regression', help='classifier which generated the results')
    parser.add_argument('-a', '--aggregate', dest='aggregate', action='store_true', help='aggregate results on sizes')
    parser.add_argument('-s', '--sizes', dest='sizes', type=int, nargs='+',
                        help='filter experiments whose training sets were of the specified size (or close to them)',
                        default=[])
    parser.add_argument('-n', '--number-of-jobs', dest='n_jobs', type=int, help='number of jobs to use', default=10)
    parser.add_argument('--cost-structure', dest='cost_structure', type=int, choices=(1, 2, 3), help='Use cost structure 1, 2 or 3', default=1)
    parser.add_argument('--load-path', dest='load_path', type=str, default='.data/active_learning/results',
                        help='path for loading data, defaults to .data/active_learning/results')
    parser.add_argument('--save-path', dest='save_path', type=str, default='.data/active_learning/minecore',
                        help='path for saving analysis, defaults to .data/active_learning/minecore')
    parser.add_argument('--save-contingency', action='store_true', help='save contingency matrices only')

    args = parser.parse_args()
    if args.sizes:
        warnings.warn("--sizes has been specified but at the moment it has no effect", UserWarning)
    policy = ALPolicy.from_string(args.policy)
    os.makedirs(args.save_path, exist_ok=True)

    if args.cost_structure == 1:
        cost_structure = cost_structure_1
    elif args.cost_structure == 2:
        cost_structure = cost_structure_2
    else:
        cost_structure = cost_structure_3

    with concurrent.futures.ProcessPoolExecutor(args.n_jobs) as p:
        futures = []
        filtered = filter(lambda m: m, map(lambda f: filter_file(f, policy, args.classifier, args.sizes, labels=None), os.listdir(args.load_path)))
        label_files = {}
        for m in filtered:
            lf = label_files.setdefault(m.group('label'), {})
            lf[int(m.group('size'))] = os.path.join(args.load_path, m.string)
        pbar = tqdm(pairs)
        for pair in pbar:
            pbar.set_description(f'Sending {pair} to the pool')
            futures.append(p.submit(minecore_costs_on_pair, pair, label_files, policy is ALPolicy.PASSIVE_LEARNING))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing and saving results'):
            results, pair = future.result()
            if type(results) is str or args.save_contingency:
                print(results)
                continue
            multindex_results = {}
            for outer_key, inner_dict in results.items():
                for inner_key, values in inner_dict.items():
                    multindex_results[(outer_key, inner_key)] = values
            df = pd.DataFrame(multindex_results).T
            if args.aggregate:
                df = df.groupby(level=1).mean()
            df.to_csv(os.path.join(args.save_path, f'{policy}_{args.classifier.replace(" ", "")}_minecore_{"-".join(pair)}_cs{args.cost_structure}.tsv'), sep='\t')
