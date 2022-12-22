import pickle
import os
import numpy as np
import re
from typing import Iterable, Optional, Set, Generator, Callable
from utils.data_utils import ALPolicy
from dataclasses import dataclass
from datetime import datetime


pattern = re.compile(r'(?P<policy>(ALvRS|ALvUS|ALvDS|ALvRUS))_idxs_checkpoint-(?P<checkpoint>\d+)_pool(?P<pool>\d+)_(?P<timestamp>\d+).pkl')


@dataclass
class ALFileInfo:
    policy: ALPolicy
    checkpoint: int
    timestamp: int
    pool_size: int
    filename: str


@dataclass
class ActiveLearningDataset:
    train_idxs: np.ndarray
    pool_test_idxs: np.ndarray
    info: ALFileInfo
    label: str


def load_AL_data(policy: ALPolicy, labels: Iterable[str], pool_idxs: Set[int],
                 path='.data/active_learning', sizes: Optional[Set[int]] = None,
                 date: Optional[datetime] = None) -> Generator[ActiveLearningDataset, None, None]:
    assert isinstance(policy, ALPolicy), f"policy should be an instance of ALPolicy, got {type(policy)} instead"
    filters = [
        lambda f: f.policy == policy,
        lambda f: f.pool_size == len(pool_idxs),
        lambda f: f.checkpoint is None or f.checkpoint in sizes,
        lambda f: date is None or f.timestamp > int(date.timestamp())
    ]
    for label in labels:
        files = map(lambda fname: _get_info_from_filename(fname), os.listdir(os.path.join(path, label)))
        for file in filter_files(files, filters):
            with open(os.path.join(path, label, file.filename), 'rb') as f:
                train_idxs = pickle.load(f)
                test_idxs = np.array(list(pool_idxs - set(train_idxs)))
                yield ActiveLearningDataset(train_idxs, test_idxs, file, label)


def filter_files(files: Iterable[ALFileInfo], filters: Iterable[Callable[[ALFileInfo], bool]]):
    return filter(lambda f: all(fl(f) for fl in filters) if f is not None else False, files)


def _get_info_from_filename(filename: str) -> Optional[ALFileInfo]:
    match = pattern.match(filename)
    if match is not None:
        return ALFileInfo(ALPolicy.from_string(match.group('policy')), int(match.group('checkpoint')),
                          int(match.group('timestamp')), int(match.group('pool')), filename)
