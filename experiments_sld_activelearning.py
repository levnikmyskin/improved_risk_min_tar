from active_learning import load_AL_data
from utils.data_utils import ALPolicy
from sklearn.datasets import fetch_rcv1
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from minecore import pairs
from utils import random_dataset_with_given_prevalences, flatten
from sld.sld import run_sld
from tqdm import tqdm
import numpy as np
import argparse
import os
import pickle
import concurrent.futures


rcv1 = fetch_rcv1()


def get_data_on_idxs(tr_idxs, te_idxs):
    return rcv1.data[tr_idxs], np.asarray(rcv1.target[tr_idxs].todense()).squeeze(), \
           rcv1.data[te_idxs], np.asarray(rcv1.target[te_idxs].todense()).squeeze()


def get_predictions(classifier_type, x_tr, y_tr, x_te):
    if classifier_type == 'Logistic Regression':
        clf = CalibratedClassifierCV(LogisticRegression(), cv=min(y_tr.sum(), 10), ensemble=False)
    else:
        clf = CalibratedClassifierCV(LinearSVC(), cv=min(y_tr.sum(), 10), ensemble=False)
    clf.fit(x_tr, y_tr)
    return clf.predict_proba(x_te), clf


def apply_sld(probs, y_tr, y_te):
    priors = np.array([1 - y_tr.mean(), y_tr.mean()])
    sld_posteriors, sld_priors, _ = run_sld(y_te, np.copy(probs), priors)
    return sld_priors, sld_posteriors


def run_experiment_for_label(dataset, args):
    al_x_tr, al_y_tr, al_x_te, al_y_te = get_data_on_idxs(dataset.train_idxs, dataset.pool_test_idxs)
    label_idx = list(rcv1.target_names).index(dataset.label)
    al_y_tr, al_y_te = al_y_tr[:, label_idx], al_y_te[:, label_idx]
    rand_x_tr, rand_y_tr, rand_x_te, rand_y_te = random_dataset_with_given_prevalences(
        rcv1.data[pool_idxs],
        np.asarray(rcv1.target[pool_idxs, label_idx].todense()).squeeze(),
        al_y_tr.mean(),
        al_y_te.mean(),
        al_y_tr.shape[0],
        al_y_te.shape[0],
        seed=args.seed
    )
    al_probs, al_clf = get_predictions(args.classifier, al_x_tr, al_y_tr, al_x_te)
    al_sld_priors, al_sld_posteriors = apply_sld(al_probs, al_y_tr, al_y_te)

    rand_probs, rand_clf = get_predictions(args.classifier, rand_x_tr, rand_y_tr, rand_x_te)
    rand_sld_priors, rand_sld_posteriors = apply_sld(rand_probs, rand_y_tr, rand_y_te)

    al_np_probs = al_clf.predict_proba(npool_x)
    rand_np_probs = rand_clf.predict_proba(npool_x)

    with open(os.path.join(args.save_path, f'{dataset.info.policy}_{dataset.label}_{dataset.info.checkpoint}size_{args.classifier.replace(" ", "")}_results.pkl'), 'wb') as f:
        data = {
            'al_y_tr': al_y_tr,
            'al_probs': al_probs,
            'al_sld_priors': al_sld_priors,
            'al_sld_posteriors': al_sld_posteriors,
            'al_y_te': al_y_te,
            'rand_y_tr': rand_y_tr,
            'rand_probs': rand_probs,
            'rand_sld_priors': rand_sld_priors,
            'rand_sld_posteriors': rand_sld_posteriors,
            'rand_y_te': rand_y_te,
            'al_np_probs': al_np_probs,
            'rand_np_probs': rand_np_probs,
            'np_y_te': npool_y[:, label_idx]
        }
        pickle.dump(data, f)


if __name__ == '__main__':
    from sklearn.exceptions import ConvergenceWarning
    from datetime import datetime
    import warnings
    parser = argparse.ArgumentParser(description="Run SLD experiments on Active Learning datasets.")
    parser.add_argument('-p', '--policy', dest='policy', choices=('RS', 'US', 'DS', 'RUS'), help='policy which generated AL datasets')
    parser.add_argument('-s', '--sizes', dest='sizes', type=int, nargs='+',
                        help='load datasets of the specified sizes only', default=[])
    parser.add_argument('-c', '--classifier', dest='classifier', choices=('Logistic Regression', 'SVM'),
                        default='Logistic Regression')
    parser.add_argument('-t', '--timestamp', type=int, help='only process datasets created after this timestamp')
    parser.add_argument('-d', '--date', help='only process datasets created after this date %d/%m[/%yyyy] (eg. 20/4/2022)')
    parser.add_argument('-n', '--number-of-jobs', dest='n_jobs', type=int, help='number of jobs to use',
                        default=30)
    parser.add_argument('--pool-size', dest='pool_size', type=int, help='size of AL pool', default=100_000)
    parser.add_argument('--seed', dest='seed', type=int, help='seed for random dataset', default=42)
    parser.add_argument('--labels', choices=('minecore', 'all', 'all_no_mc'), help='either "minecore", "all" or "all_no_mc". Former will run on original Minecore pairs, latter on all labels except original Minecore\'s', default='minecore')
    parser.add_argument('--save-path', dest='save_path', type=str, help='OS path to save experiment results', default='')
    parser.add_argument('--load-path', dest='load_path', type=str, default='.data/active_learning',
                        help='path for loading data, defaults to .data/active_learning')

    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    args = parser.parse_args()
    policy = ALPolicy.from_string(args.policy)

    date = None
    if args.timestamp:
        date = datetime.fromtimestamp(args.timestamp)
    elif args.date:
        d = list(map(int, args.date.split('/')))
        if len(d) < 2 or len(d) > 3:
            warnings.warn(f"Invalid date format passed. This is undefined behaviour. Required: %d/%m[/%yyyy], got {args.date}")
        date = datetime(year=d[-1] if len(d) == 3 else datetime.today().year, month=d[1], day=d[0])
    else:
        date = datetime.fromtimestamp(0)

    os.makedirs(args.save_path, exist_ok=True)
    pool_idxs = np.arange(args.pool_size)
    non_pool_idxs = np.arange(args.pool_size, args.pool_size*2)
    npool_x, npool_y = rcv1.data[non_pool_idxs], np.asarray(rcv1.target[non_pool_idxs].todense()).squeeze()
    if args.labels == 'minecore':
        labels = set(flatten(pairs))
    elif args.labels == 'all':
        labels = rcv1.target_names.tolist()
    else:
        labels = set(rcv1.target_names.tolist()) - set(flatten(pairs))

    with concurrent.futures.ProcessPoolExecutor(args.n_jobs) as p:
        futures = []
        for dataset in tqdm(load_AL_data(policy, labels, set(pool_idxs), args.load_path, sizes=args.sizes, date=date),
                            desc='Submitting jobs to the pool'):
            futures.append(p.submit(run_experiment_for_label, dataset, args))

        for future in tqdm(concurrent.futures.as_completed(futures), desc='Running experiments', total=len(futures)):
            future.result()
