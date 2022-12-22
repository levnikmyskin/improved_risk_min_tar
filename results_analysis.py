import argparse
import functools
import pickle
import numpy as np
import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm
from minecore import pairs
from utils.data_utils import ALPolicy, flatten
from utils import smoothmacroF1, aggregate_same_sizes, filter_file
from utils.calibration_error_metrics import ACELoss, ECELoss
from sklearn.metrics import recall_score, precision_score, accuracy_score, brier_score_loss, f1_score
from utils.misc import AlwaysIn, nested_dict_to_pandas


def recall_with_probs(y, probs):
    return recall_score(y, (probs > .5).astype(int))


def precision_with_probs(y, probs):
    return precision_score(y, (probs > .5).astype(int))


def accuracy_with_probs(y, probs):
    return accuracy_score(y, (probs > .5).astype(int))


def f1_with_probs(y, probs):
    return f1_score(y, (probs > .5).astype(int))


def ace(y, probs):
    probs = np.expand_dims(probs, 1)
    probs = np.concatenate((1 - probs, probs), axis=1)
    return ACELoss().loss(probs, y, logits=False)


def ece(y, probs):
    probs = np.expand_dims(probs, 1)
    probs = np.concatenate((1 - probs, probs), axis=1)
    return ECELoss().loss(probs, y, logits=False)


def _compute_metrics(y, probs, metrics):
    return {
        k: v(y, probs.copy()) for k, v in metrics.items()
    }


def one_minus_brier(*args, **kwargs):
    return 1 - brier_score_loss(*args, **kwargs)


def compute_metrics_on_results(filename, metrics):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    if args.compute_metrics_on_all:
        al_y_metric = np.concatenate((results['al_y_tr'], results['al_y_te']))
        rand_y_metric = np.concatenate((results['rand_y_tr'], results['rand_y_te']))
        al_probs_tr = results['al_y_tr']
        rand_probs_tr = results['rand_y_tr']
        al_mle_probs = np.concatenate((al_probs_tr, results['al_probs'][:, 1]))
        rand_mle_probs = np.concatenate((rand_probs_tr, results['rand_probs'][:, 1]))
        al_sld_probs = np.concatenate((al_probs_tr, results['al_sld_posteriors'][:, 1]))
        rand_sld_probs = np.concatenate((rand_probs_tr, results['rand_sld_posteriors'][:, 1]))
    elif args.compute_metrics_on_tr:
        al_y_metric = np.concatenate((results['al_y_tr'], results['al_y_te']))
        rand_y_metric = np.concatenate((results['rand_y_tr'], results['rand_y_te']))
        al_probs_tr = results['al_y_tr']
        rand_probs_tr = results['rand_y_tr']
        al_mle_probs = np.concatenate((al_probs_tr, np.zeros_like(results['al_probs'][:, 1])))
        rand_mle_probs = np.concatenate((rand_probs_tr, np.zeros_like(results['rand_probs'][:, 1])))
        al_sld_probs = np.concatenate((al_probs_tr, np.zeros_like(results['al_sld_posteriors'][:, 1])))
        rand_sld_probs = np.concatenate((rand_probs_tr, np.zeros_like(results['rand_sld_posteriors'][:, 1])))
    else:
        al_y_metric = results['al_y_te']
        rand_y_metric = results['rand_y_te']
        al_mle_probs = results['al_probs'][:, 1]
        rand_mle_probs = results['rand_probs'][:, 1]
        al_sld_probs = results['al_sld_posteriors'][:, 1]
        rand_sld_probs = results['rand_sld_posteriors'][:, 1]
    return pd.DataFrame({
        'AL MLE': _compute_metrics(al_y_metric, al_mle_probs, metrics),
        'AL SLD': _compute_metrics(al_y_metric, al_sld_probs, metrics),
        'Rand MLE': _compute_metrics(rand_y_metric, rand_mle_probs, metrics),
        'Rand SLD': _compute_metrics(rand_y_metric, rand_sld_probs, metrics),
        'AL NP MLE': _compute_metrics(results['np_y_te'], results['al_np_probs'][:, 1], metrics),
        'Rand NP MLE': _compute_metrics(results['np_y_te'], results['rand_np_probs'][:, 1], metrics),
        'Tr prev.': results['al_y_tr'].mean(),
        'Te prev.': results['al_y_te'].mean(),
        'NP prev.': results['np_y_te'].mean()
    }), filename


def prepare_for_latex(df_m, df_s):
    for col in df_m.columns:
        df_m[col] = df_m[col].transform(lambda x: round(x, 3)).astype(str) + ' $\pm$ ' + df_s[col].transform(lambda x: round(x, 3)).astype(str)
    return df_m


# This function is here only because pool won't be able to pickle it otherwise (we cannot use lambda in pool.map)
def elaborate_on_file(file, metrics, load_path):
    return compute_metrics_on_results(os.path.join(load_path, file), metrics)[0]


def compute_aggregated_metrics(files, size, policy, classifier, metrics, load_path, pool, labels_str, return_standard_df=False):
    class FutureMock:
        def __init__(self, df, fname):
            self.df = df
            self.fname = fname

        def result(self):
            return self.df, self.fname

    results = pd.concat(pool.map(functools.partial(elaborate_on_file, metrics=metrics, load_path=load_path), files, chunksize=max(1, len(files) // args.n_jobs)))
    res_mean = results.groupby(results.index).mean()
    res_std = results.groupby(results.index).std()
    if return_standard_df:
        return FutureMock(res_mean, '')
    res_mean = prepare_for_latex(res_mean, res_std)
    return FutureMock(res_mean, f"{policy}_{size}_aggregated-metrics_{classifier.replace(' ', '')}_{labels_str}-labels.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run several analysis and evaluations on experiments results")
    parser.add_argument('-p', '--policy', dest='policy', choices=('RS', 'US', 'RUS'), help='policy which generated results')
    parser.add_argument('-s', '--sizes', dest='sizes', type=int, nargs='+',
                        help='filter experiments whose training sets were of the specified size (or close to them)', default=[])
    parser.add_argument('-c', '--classifier', dest='classifier', choices=('Logistic Regression', 'SVM'),
                        default='Logistic Regression', help='classifier which generated the results')
    parser.add_argument('-m', '--metrics', dest='metrics',
                        choices=('sf1', 'f1', 'precision', 'accuracy', 'recall', 'brier', '1-brier', 'ace', 'ece'), nargs='+',
                        help='metrics to compute on results')
    parser.add_argument('-a', '--aggregate', dest='aggregate', action='store_true', help='aggregate results for same '
                                                                                         'sizes')
    parser.add_argument('-n', '--number-of-jobs', dest='n_jobs', type=int, help='number of jobs to use', default=10)
    parser.add_argument('--average-all-sizes', dest='avg_sizes', action='store_true', help='average metrics on all different sizes')
    parser.add_argument('--labels', choices=('minecore', 'all'), help='either "minecore" or "all". Former will run on original Minecore pairs.', default='minecore')
    parser.add_argument('--load-path', dest='load_path', type=str, default='.data/active_learning/results',
                        help='path for loading data, defaults to .data/active_learning/results')
    parser.add_argument('--save-path', dest='save_path', type=str, default='.data/active_learning/metrics',
                        help='path for saving analysis, defaults to .data/active_learning/metrics')
    parser.add_argument('--compute-metrics-on-all', action='store_true', help='compute metrics on predictions on all data')
    parser.add_argument('--compute-metrics-on-tr', action='store_true', help='compute metrics on training set')
    parser.add_argument('--dry-run', action='store_true', help='run without saving')

    metrics_mapping = {
        'sf1': smoothmacroF1,
        'f1': f1_with_probs,
        'precision': precision_with_probs,
        'accuracy': accuracy_with_probs,
        'recall': recall_with_probs,
        'brier': brier_score_loss,
        '1-brier': one_minus_brier,
        'ace': ace,
        'ece': ece
    }
    args = parser.parse_args()
    policy = ALPolicy.from_string(args.policy)

    if args.labels == 'minecore':
        labels = set(flatten(pairs))
    else:
        # No point in loading RCV1 and creating a set of all classes. We simply have a container that always
        # outputs True to the statement `l in labels`
        labels = AlwaysIn()
    os.makedirs(args.save_path, exist_ok=True)
    metrics = {k: v for k, v in metrics_mapping.items() if k in args.metrics}
    with concurrent.futures.ProcessPoolExecutor(args.n_jobs) as p:
        futures = []
        if not args.aggregate:
            for filename in tqdm(filter(lambda f: filter_file(f, policy, args.classifier, set(args.sizes), labels), os.listdir(args.load_path))):
                futures.append(p.submit(compute_metrics_on_results, os.path.join(args.load_path, filename), metrics))
            iterator = concurrent.futures.as_completed(futures)
            it_len = len(futures)
        else:
            aggregated_files = aggregate_same_sizes(policy, args.classifier, os.listdir(args.load_path), set(args.sizes), labels)
            iterator = []
            for size, file_lists in tqdm(aggregated_files.items(), desc='Computing aggregated metrics'):
                iterator.append(compute_aggregated_metrics(file_lists, size, policy, args.classifier, metrics, args.load_path, p, args.labels, args.avg_sizes))
            it_len = len(iterator)

        avg_dfs = None
        for future in tqdm(iterator, desc='Saving analysis', total=it_len):
            df, filename = future.result()
            size = int(filename.split('_')[1])
            if args.avg_sizes:
                avg_dfs = pd.concat((avg_dfs, df))
            elif avg_dfs is None:
                if 'recall' in args.metrics:
                    avg_dfs = {size: df[['AL MLE', 'Tr prev.', 'Te prev.']].reset_index().drop('index', axis=1).
                        rename(mapper={'AL MLE': 'recall'}, axis=1).T.to_dict()}
                elif 'ace' in args.metrics or 'ece' in args.metrics:
                    avg_dfs = {size: df[['AL MLE', 'Rand MLE']].to_dict()}
            else:
                if 'recall' in args.metrics:
                    avg_dfs[size] = df[['AL MLE', 'Tr prev.', 'Te prev.']].reset_index().drop('index', axis=1).\
                        rename(mapper={'AL MLE': 'recall'}, axis=1).T.to_dict()
                else:
                    avg_dfs[size] = df[['AL MLE', 'Rand MLE']].to_dict()
            if args.dry_run:
                continue
            else:
                filename = os.path.basename(filename)
                df.to_csv(os.path.join(args.save_path, filename.replace('.pkl', '.tsv')), sep='\t')

        if args.avg_sizes:
            group = avg_dfs.groupby(avg_dfs.index)
            avg_dfs = prepare_for_latex(group.mean(), group.std())
            if not args.dry_run:
                avg_dfs.to_csv(os.path.join(args.save_path, f'{policy}_avg_aggregated-metrics_{args.classifier.replace(" ", "")}_{args.labels}-labels.tsv'), sep='\t')
        else:
            if policy is ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING:
                avg_dfs = nested_dict_to_pandas(avg_dfs).droplevel(1, 1).T.sort_index()
            else:
                avg_dfs = nested_dict_to_pandas(avg_dfs).droplevel(1, 1).T.drop(1000).sort_index()
            print(avg_dfs.to_latex(escape=False))
