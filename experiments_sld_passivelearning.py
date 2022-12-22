from experiments_sld_activelearning import get_predictions, apply_sld
from sklearn.datasets import fetch_rcv1
from utils.data_utils import random_sample_from_dataset
from utils import flatten
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from minecore import pairs
from tqdm import tqdm
from typing import Optional
import numpy as np
import argparse
import os
import pickle
import concurrent.futures


rcv1 = fetch_rcv1()


def run_passive_learning_on_size(label: str, tr_size: int, pool_size: int, classifier: str, seed: Optional[int],
                                 save_path: str):
    label_idx = rcv1.target_names.tolist().index(label)
    x, y = rcv1.data[:pool_size], rcv1.target[:pool_size][:, label_idx]
    x_tr, y_tr, x_te, y_te = random_sample_from_dataset(x, y, tr_size, seed=seed)
    y_tr, y_te = y_tr.toarray().squeeze(), y_te.toarray().squeeze()
    probs, clf = get_predictions(classifier, x_tr, y_tr, x_te)
    sld_priors, sld_post = apply_sld(probs, y_tr, y_te)
    np_probs = clf.predict_proba(npool_x)
    with open(os.path.join(save_path, f'PL_{label}_{tr_size}size_{classifier.replace(" ", "")}_results.pkl'), 'wb') as f:
        pickle.dump({
            'y_tr': y_tr,
            'y_te': y_te,
            'np_y_te': npool_y[:, label_idx].squeeze(),
            'mle_probs': probs,
            'sld_priors': sld_priors,
            'sld_post': sld_post,
            'np_probs': np_probs
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SLD experiments on Passive Learning using RCV1')
    parser.add_argument('-s', '--sizes', dest='sizes', type=int, nargs='+', help='Run experiments on these training sizes')
    parser.add_argument('-c', '--classifier', dest='classifier', choices=('Logistic Regression', 'SVM'),
                        default='Logistic Regression')
    parser.add_argument('-n', '--number-of-jobs', dest='n_jobs', type=int, help='number of jobs to use',
                        default=30)
    parser.add_argument('--seed', type=int, default=42, help='seed for RNG')
    parser.add_argument('--pool-size', dest='pool_size', type=int, help='size for random sampling', default=100_000)
    parser.add_argument('--save-path', dest='save_path', type=str, help='OS path to save experiment results', default='')

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    pool_idxs = np.arange(args.pool_size)
    non_pool_idxs = np.arange(args.pool_size, args.pool_size*2)
    npool_x, npool_y = rcv1.data[non_pool_idxs], np.asarray(rcv1.target[non_pool_idxs].todense()).squeeze()

    with concurrent.futures.ProcessPoolExecutor(args.n_jobs) as p:
        futures = []
        for label in flatten(set(pairs)):
            for size in args.sizes:
                futures.append(p.submit(run_passive_learning_on_size, label, size, args.pool_size, args.classifier,
                                        args.seed, args.save_path))

        for future in tqdm(concurrent.futures.as_completed(futures), desc='Running experiments', total=len(futures)):
            future.result()
