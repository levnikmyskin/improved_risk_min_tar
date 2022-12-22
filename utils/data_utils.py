import enum
import itertools
import re
from typing import Optional, Set, Match
import numpy as np


pattern = re.compile(r'(?P<policy>(ALvUS|ALvRS|PL|ALvRUS))_(?P<label>.+)_(?P<size>\d+)size_(?P<classifier>.+)_results\.pkl')


class SLDConst(enum.Enum):
    RS_SLD = enum.auto()
    US_SLD = enum.auto()
    PL_SLD = enum.auto()
    RUS_SLD = enum.auto()
    RANDRS_SLD = enum.auto()
    RANDUS_SLD = enum.auto()
    RANDRUS_SLD = enum.auto()

    def __str__(self):
        if self is self.RS_SLD:
            return "RS + SLD"
        elif self is self.US_SLD:
            return "US + SLD"
        elif self is self.PL_SLD:
            return "PL + SLD"
        elif self is self.RUS_SLD:
            return "RUS + SLD"
        elif self is self.RANDRS_SLD:
            return "RAND RS + SLD"
        elif self is self.RANDUS_SLD:
            return "RAND US + SLD"
        elif self is self.RANDRUS_SLD:
            return "RAND RUS + SLD"

    @staticmethod
    def from_al_policy(policy: 'ALPolicy'):
        if policy is ALPolicy.RELEVANCE_SAMPLING:
            return SLDConst.RS_SLD
        elif policy is ALPolicy.UNCERTAINTY_SAMPLING:
            return SLDConst.US_SLD
        elif policy is ALPolicy.PASSIVE_LEARNING:
            return SLDConst.PL_SLD
        elif policy is ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING:
            return SLDConst.RUS_SLD
        elif policy is ALPolicy.RAND_RS:
            return SLDConst.RANDRS_SLD
        elif policy is ALPolicy.RAND_US:
            return SLDConst.RANDUS_SLD
        elif policy is ALPolicy.RAND_RUS:
            return SLDConst.RANDRUS_SLD


class ALPolicy(enum.IntEnum):
    RELEVANCE_SAMPLING = 1
    UNCERTAINTY_SAMPLING = 2
    PASSIVE_LEARNING = 0
    QUIRE_SAMPLING = 4
    DIVERSITY_SAMPLING = 5
    HIERARCHICAL_SAMPLING = 6
    RAND_RS = 7
    RAND_US = 8
    RELEVANCE_UNCERTAINTY_SAMPLING = 9
    RAND_RUS = 10

    @staticmethod
    def from_string(string: str) -> 'ALPolicy':
        if string == 'ALvRS' or string == 'RS':
            return ALPolicy.RELEVANCE_SAMPLING
        elif string == 'ALvUS' or string == 'US':
            return ALPolicy.UNCERTAINTY_SAMPLING
        elif string == 'ALvQUIRE' or string == 'QUIRE':
            return ALPolicy.QUIRE_SAMPLING
        elif string == 'ALvDS' or string == 'DS':
            return ALPolicy.DIVERSITY_SAMPLING
        elif string == 'ALvHS' or string == 'HS':
            return ALPolicy.HIERARCHICAL_SAMPLING
        elif string == 'PL':
            return ALPolicy.PASSIVE_LEARNING
        elif string == 'RANDRS' or string == 'Rand (RS)':
            return ALPolicy.RAND_RS
        elif string == 'RANDUS' or string == 'Rand (US)':
            return ALPolicy.RAND_US
        elif string == 'ALvRUS' or string == 'RUS':
            return ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING
        elif string == 'RANDRUS' or string == 'Rand (RUS)':
            return ALPolicy.RAND_RUS

    def compact_str(self):
        if self is self.RELEVANCE_SAMPLING:
            return "RS"
        elif self is self.UNCERTAINTY_SAMPLING:
            return "US"
        elif self is self.QUIRE_SAMPLING:
            return "QUIRE"
        elif self is self.DIVERSITY_SAMPLING:
            return "DS"
        elif self is self.HIERARCHICAL_SAMPLING:
            return "HS"
        elif self is self.RELEVANCE_UNCERTAINTY_SAMPLING:
            return "RUS"
        else:
            return str(self)

    def is_self_with_sld(self, sld_const: SLDConst) -> bool:
        if self is self.RELEVANCE_SAMPLING and sld_const is SLDConst.RS_SLD:
            return True
        elif self is self.UNCERTAINTY_SAMPLING and sld_const is SLDConst.US_SLD:
            return True
        elif self is self.PASSIVE_LEARNING and sld_const is SLDConst.PL_SLD:
            return True
        elif self is self.RAND_RS and sld_const is SLDConst.RANDRS_SLD:
            return True
        elif self is self.RAND_US and sld_const is SLDConst.RANDUS_SLD:
            return True
        elif self is self.RELEVANCE_UNCERTAINTY_SAMPLING and sld_const is SLDConst.RUS_SLD:
            return True
        elif self is self.RAND_RUS and sld_const is SLDConst.RANDRUS_SLD:
            return True
        return False

    def to_rand(self):
        if self is self.UNCERTAINTY_SAMPLING:
            return ALPolicy.RAND_US
        elif self is self.RELEVANCE_SAMPLING:
            return ALPolicy.RAND_RS
        elif self is self.RELEVANCE_UNCERTAINTY_SAMPLING:
            return ALPolicy.RAND_RUS
        raise ValueError(f'policy {self} does not have a Rand equivalent')

    def __str__(self):
        if self is self.RELEVANCE_SAMPLING:
            return "ALvRS"
        elif self is self.UNCERTAINTY_SAMPLING:
            return "ALvUS"
        elif self is self.QUIRE_SAMPLING:
            return "ALvQUIRE"
        elif self is self.DIVERSITY_SAMPLING:
            return "ALvDS"
        elif self is self.HIERARCHICAL_SAMPLING:
            return "ALvHS"
        elif self is self.PASSIVE_LEARNING:
            return "PL"
        elif self is self.RAND_RS:
            return "Rand (RS)"
        elif self is self.RAND_US:
            return "Rand (US)"
        elif self is self.RELEVANCE_UNCERTAINTY_SAMPLING:
            return "ALvRUS"
        elif self is self.RAND_RUS:
            return "Rand (RUS)"


def random_dataset_with_given_prevalences(x, y, tr_prev, te_prev, tr_size, te_size, seed=None):
    tr_pos_to_take = round(tr_size * tr_prev)
    tr_neg_to_take = abs(tr_size - tr_pos_to_take)
    te_pos_to_take = round(te_size * te_prev)
    te_neg_to_take = abs(te_pos_to_take - te_size)

    x_tr, y_tr, idxs = get_xy_with_given_pos_and_negs(x, y, tr_pos_to_take, tr_neg_to_take, seed)

    te_indices = set(np.arange(y.shape[0])) - set(idxs)
    x_te, y_te = x[list(te_indices)], y[list(te_indices)]
    x_te, y_te, _ = get_xy_with_given_pos_and_negs(x_te, y_te, te_pos_to_take, te_neg_to_take, seed)
    return x_tr, y_tr, x_te, y_te


def random_sample_from_dataset(x, y, tr_size: int, seed=None):
    indices = np.arange(y.shape[0])
    training_idxs = np.random.default_rng(seed=seed).choice(indices, size=tr_size, replace=False)
    test_idxs = list(set(indices) - set(training_idxs))
    return x[training_idxs], y[training_idxs], x[test_idxs], y[test_idxs]


def get_xy_with_given_pos_and_negs(x, y, pos_to_take, neg_to_take, seed):
    positives = np.random.default_rng(seed=seed).choice(np.where(y == 1)[0], size=pos_to_take, replace=False)
    negatives = np.random.default_rng(seed=seed).choice(np.where(y == 0)[0], size=neg_to_take, replace=False)
    idxs = np.hstack((positives, negatives))
    np.random.default_rng().shuffle(idxs)
    return x[idxs], y[idxs], idxs


def flatten(list_of_lists):
    # Flatten one level of nesting. See https://docs.python.org/3.6/library/itertools.html#recipes
    return itertools.chain.from_iterable(list_of_lists)


def take(n, iterable):
    # Return first n items of the iterable as a list
    return list(itertools.islice(iterable, n))


def filter_file(filename: str, policy: Optional[ALPolicy], classifier: Optional[str], sizes: Optional[Set[int]],
                labels: Optional[Set[int]]) -> Optional[Match]:
    match = pattern.match(filename)
    if not match:
        return None
    if policy and ALPolicy.from_string(match.group('policy')) is not policy:
        return None
    if classifier and match.group('classifier') != classifier.replace(' ', ''):
        return None
    if sizes and int(match.group('size')) not in sizes:
        return None
    if labels and match.group('label') not in labels:
        return None

    return match


def aggregate_same_sizes(policy, classifier, file_list, sizes, labels):
    aggregated = {}
    for file in file_list:
        match = filter_file(file, policy, classifier, sizes, labels)
        if not match:
            continue
        size_f = aggregated.setdefault(int(match.group('size')), [])
        size_f.append(file)
    return aggregated

