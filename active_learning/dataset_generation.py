from datetime import datetime
from utils.data_utils import take, ALPolicy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from typing import Iterable, Optional
from active_learning.kmeans_diversity import DiversityWithKMeans
import enum
import numpy as np
import os.path
import pickle
import itertools
import warnings


class InitialSeedPolicy(enum.Enum):
    DETERMINISTIC = enum.auto()
    RANDOM = enum.auto()


class ActiveLearningDatasetGenerator:

    def __init__(self, policy: ALPolicy, x, y, initial_seed: int, seed_policy: InitialSeedPolicy, checkpoints: Iterable[int],
                 retain_inital_seed=True, save_path='', random_initial_pos=1, kmeans: Optional[DiversityWithKMeans] = None):
        assert isinstance(policy, ALPolicy), f"policy should be an instance of ALPolicy enum class, found {type(policy)}"
        assert isinstance(seed_policy, InitialSeedPolicy), f"seed_policy should be an instance of InitialSeedPolicy class, found {type(seed_policy)}"
        if policy is ALPolicy.DIVERSITY_SAMPLING:
            assert kmeans is not None, f'If {policy} is selected, you must provide kmeans'
        self.policy = policy
        self.x = x
        self.y = y
        self.initial_seed = initial_seed
        self.seed_policy = seed_policy
        self.checkpoints = checkpoints
        self.sorted_checkpoints = sorted(self.checkpoints)
        self.retain_initial_seed = retain_inital_seed
        self.save_path = save_path
        self.random_initial_pos = random_initial_pos
        self.rng = np.random.default_rng(seed=42)
        self.kmeans = kmeans
        self.iteration = 0
        if self.policy is ALPolicy.HIERARCHICAL_SAMPLING or self.policy is ALPolicy.DIVERSITY_SAMPLING:
            warnings.warn(f"{self.policy} was selected. `predict_on_idxs` will do nothing and return an empty array")
            # libact Dataset wants an Y array where all unlabeled instances are None
            # dataset = Dataset(x, np.empty_like(y, dtype=object))
            # self._sub_qs = UncertaintySampling(dataset, method='lc')
            # self._hs = HierarchicalSampling(dataset=dataset, classes=[0, 1], active_selecting=True, subsample_qs=self._sub_qs)

    def generate_dataset(self, dataset_size: int, kfold=10, batch_size=1000, pool_size=100_000):
        train_idx_set = set()
        if self.seed_policy is InitialSeedPolicy.DETERMINISTIC:
            train_idxs, _ = self.__deterministic_initial_seed()
        else:
            train_idxs, _ = self.__random_initial_seed()
        if self.retain_initial_seed:
            train_idx_set.update(train_idxs)
        predictions = self.predict_on_idxs(train_idxs, kfold)
        current_size = 1000
        self.__save_dataset(list(train_idx_set), len(train_idx_set), pool_size)
        while len(train_idx_set) < dataset_size:
            if self.__need_different_batch_to_next_checkpoint(current_size, batch_size):
                to_add = min(batch_size, self.__next_checkpoint(current_size, batch_size) - current_size)
            else:
                to_add = min(batch_size, dataset_size - len(train_idx_set))
            current_size += to_add
            if self.policy is ALPolicy.DIVERSITY_SAMPLING:
                self.kmeans.batch_size = to_add
                test_idxs = list(set(np.arange(len(self.y))) - train_idx_set)
                al_idxs = self.__idxs_via_policy(predictions, test_idxs)
            elif self.policy is ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING:
                al_idxs = self.__idxs_via_policy(predictions)
                rs_idxs = take(to_add // 2, filter(lambda i: i not in train_idx_set, al_idxs[:len(al_idxs) // 2]))
                train_idx_set.update(rs_idxs)
                to_add = (to_add // 2) + (to_add % 2)
                us_idxs = take(to_add, filter(lambda i: i not in train_idx_set, al_idxs[len(al_idxs) // 2:]))
                al_idxs = rs_idxs + us_idxs
            else:
                al_idxs = self.__idxs_via_policy(predictions)
                al_idxs = take(to_add, filter(lambda i: i not in train_idx_set, al_idxs))
            train_idx_set.update(al_idxs)
            train_idx_list = list(train_idx_set)
            if len(train_idx_set) < dataset_size:
                predictions = self.predict_on_idxs(train_idx_list, kfold)
            if current_size in self.checkpoints:
                self.__save_dataset(train_idx_list, len(train_idx_set), pool_size)
            elif current_size > self.sorted_checkpoints[-1]:
                break
            self.iteration += 1

    def predict_on_idxs(self, idxs, kfold):
        if self.policy is ALPolicy.QUIRE_SAMPLING or self.policy is ALPolicy.HIERARCHICAL_SAMPLING or \
                self.policy is ALPolicy.DIVERSITY_SAMPLING:
            return np.empty_like(self.y)
        x_tr, y_tr = self.__get_idxs_data(idxs)
        classifier = CalibratedClassifierCV(LinearSVC(loss='hinge'), ensemble=False, cv=min(y_tr.sum(), kfold))
        classifier.fit(x_tr, y_tr)
        return classifier.predict_proba(self.x)[:, 1]

    def __deterministic_initial_seed(self):
        pos_idx = self.rng.choice(np.where(self.y == 1)[0], size=self.initial_seed // 2, replace=False)
        neg_idx = self.rng.choice(np.where(self.y == 0)[0], size=self.initial_seed // 2, replace=False)
        train_idx = np.hstack((pos_idx, neg_idx))

        test_mask = np.ones(self.x.shape[0], dtype=bool)
        test_mask[train_idx] = False

        return train_idx, test_mask.nonzero()

    def __random_initial_seed(self):
        pos_item = self.rng.choice(np.where(self.y == 1)[0], size=self.random_initial_pos, replace=False)
        idxs = list(set(np.arange(self.x.shape[0])) - set(pos_item))
        train_idx = np.append(self.rng.choice(idxs, self.initial_seed - len(pos_item), replace=False), pos_item)
        test_mask = np.ones(self.x.shape[0], dtype=bool)
        test_mask[train_idx] = False

        return train_idx, test_mask.nonzero()

    def __idxs_via_policy(self, probs, test_idxs=None):
        if self.policy is ALPolicy.RELEVANCE_SAMPLING:
            return np.argsort(1 - probs)
        elif self.policy is ALPolicy.UNCERTAINTY_SAMPLING:
            return np.argsort(np.abs(probs - 0.5))
        elif self.policy is ALPolicy.HIERARCHICAL_SAMPLING:
            return np.argsort(1 - probs)
        elif self.policy is ALPolicy.QUIRE_SAMPLING:
            return np.argsort(1 - probs)
        elif self.policy is ALPolicy.DIVERSITY_SAMPLING:
            return self.kmeans.get_data_to_annotate(self.x[test_idxs], np.full_like(self.y, -1, dtype=int), test_idxs)
        elif self.policy is ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING:
            rs_sort = np.argsort(1 - probs)
            us_sort = np.argsort(np.abs(probs - 0.5))
            idxs = np.concatenate((rs_sort, us_sort))
            return idxs
            # idxs = []
            # for _ in range(to_add):
            #     idx = self._quire.make_query()
            #     self._quire.update(idx, self.y[idx])
            #     idxs.append(idx)
            # return idxs

    def __get_idxs_data(self, train_idxs):
        return self.x[train_idxs], self.y[train_idxs]

    def __need_different_batch_to_next_checkpoint(self, current_size: int, batch_size: int) -> bool:
        return current_size + batch_size != self.__next_checkpoint(current_size, batch_size)

    def __next_checkpoint(self, current_size: int, batch_size: int) -> int:
        try:
            return next(itertools.dropwhile(lambda c: c <= current_size, self.sorted_checkpoints))
        except StopIteration:
            # Return the exact size that `__need_different_batch_to_next_checkpoint` expects.
            # This is a "dirty" way to say that there are no more checkpoints and AL should go on with batch_size
            return current_size + batch_size

    def __save_dataset(self, idxs, checkpoint, pool_size):
        with open(os.path.join(self.save_path, f'{self.policy}_idxs_checkpoint-{checkpoint}_pool{pool_size}_{int(datetime.now().timestamp())}.pkl'),
                  'wb') as f:
            pickle.dump(idxs, f)
