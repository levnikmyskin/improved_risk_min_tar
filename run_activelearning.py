import argparse
import os
import concurrent.futures
import numpy as np
from active_learning import ActiveLearningDatasetGenerator, InitialSeedPolicy, DiversityWithKMeans
from utils.data_utils import ALPolicy
from minecore import pairs
from sklearn.datasets import fetch_rcv1
from tqdm import tqdm
from utils import flatten


def run_dataset_generation(policy, x, y, initial_seed, init_seed_policy, checkpoints, retain_initial, save_path,
                           dataset_size, batch_size, label, pool_size, initial_seed_pos, kmeans=None):
    try:
        al = ActiveLearningDatasetGenerator(policy, x, y, initial_seed, init_seed_policy, checkpoints,
                                            retain_initial, save_path=save_path, random_initial_pos=initial_seed_pos, kmeans=kmeans)
        al.generate_dataset(dataset_size, batch_size=batch_size, pool_size=pool_size)
    except Exception as e:
        return Exception(f"Dataset generation for label {label} failed: {e}")


if __name__ == '__main__':
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    parser = argparse.ArgumentParser(description='Generate a dataset of specified size from RCV1v2 via Active Learning '
                                                 '(either via RS, US, or QUIRE).')
    parser.add_argument('-s', '--dataset-size', dest='dataset_size', type=int, help='size of the generated dataset',
                        required=True)
    parser.add_argument('-c', '--checkpoints', dest='checkpoints', type=int, nargs='+',
                        help='one or more sizes to save dataset up to that size', default=[])
    parser.add_argument('-p', '--policy', dest='policy', choices=('RS', 'US', 'QUIRE', 'HS', 'DS', 'RUS'),
                        help='policy to be used for AL',
                        required=True)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=1000, help='number of elements '
                                                                                              'to add at each '
                                                                                              'iteration')
    parser.add_argument('-i', '--initial_seed', dest='initial_seed', type=int, default=1000,
                        help='number of initial items')
    parser.add_argument('-n', '--number-of-jobs', dest='n_jobs', type=int, help='number of jobs to use',
                        default=30)
    parser.add_argument('--pool-size', dest='pool_size', type=int, help='size of AL pool', default=100_000)
    parser.add_argument('--labels', choices=('minecore', 'all', 'all_no_mc'), help='either "minecore", "all" or "all_no_mc". Former will run on original Minecore pairs, latter on all labels except original Minecore\'s', default='minecore')
    parser.add_argument('--initial_seed_type', dest='initial_seed_type', choices=('deterministic', 'random'),
                        help='specify whether initial seed items should be taken at random or not', default='random')
    parser.add_argument('--initial-seed-pos', type=int, default=1, help='if initial_seed_type is random, the number of granted positives in the initial seed')
    parser.add_argument('-r', '--no_retain_initial', dest='no_retain_initial', action='store_true', help='if specified '
                                                                                                         'initial seed '
                                                                                                         'will be discarded')
    parser.add_argument('--total-per-cluster', type=int, help='if ALvDS, total items per cluster to take', default=100)
    parser.add_argument('--centroids', type=int, help='if ALvDS, centroids per cluster to take', default=30)
    parser.add_argument('--outliers', type=int, help='if ALvDS, outliers per cluster to take', default=30)
    parser.add_argument('--outlier-perc', type=int, help='if ALvDS, percentile after which an item is considered an outlier', default=80)
    parser.add_argument('--kmeans-rand', type=int, help='if ALvDS, random items per cluster to take', default=40)
    parser.add_argument('--save_path', type=str, help='OS path to save generated dataset(s)', default='')

    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    policy = ALPolicy.from_string(args.policy)
    if policy is ALPolicy.DIVERSITY_SAMPLING:
        assert sum((args.centroids, args.outliers, args.kmeans_rand)) == args.total_per_cluster, 'kmeans sampling (centroids, outliers, random) does not sum to total_per_cluster'

    if args.initial_seed_type == 'deterministic':
        initial_seed_policy = InitialSeedPolicy.DETERMINISTIC
    else:
        initial_seed_policy = InitialSeedPolicy.RANDOM

    rcv1 = fetch_rcv1()
    if args.labels == 'minecore':
        labels = set(flatten(pairs))
    elif args.labels == 'all':
        labels = rcv1.target_names.tolist()
    else:
        labels = set(rcv1.target_names.tolist()) - set(flatten(pairs))

    x_pool, y_pool = rcv1.data[:args.pool_size], rcv1.target[:args.pool_size]
    diversity = DiversityWithKMeans(args.total_per_cluster, args.centroids, args.outliers, args.kmeans_rand, args.outlier_perc, args.batch_size)
    n_jobs = args.jobs if args.n_jobs <= len(labels) else len(labels)
    with concurrent.futures.ProcessPoolExecutor(n_jobs) as p:
        results = []
        for label in labels:
            y_label = y_pool[:, list(rcv1.target_names).index(label)]
            y_label = np.asarray(y_label.todense()).squeeze()
            save_path = os.path.join(args.save_path, label)
            os.makedirs(save_path, exist_ok=True)
            results.append(
                p.submit(run_dataset_generation, policy, x_pool, y_label, args.initial_seed, initial_seed_policy,
                         args.checkpoints, not args.no_retain_initial, save_path, args.dataset_size, args.batch_size,
                         label, args.pool_size, args.initial_seed_pos, diversity))

        for future in tqdm(concurrent.futures.as_completed(results), total=len(results)):
            res = future.result()
            if isinstance(res, Exception):
                print(res)
