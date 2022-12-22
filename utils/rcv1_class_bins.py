import numpy as np
import pandas as pd
import numpy.typing as npt
import re
from typing import List, Dict, Tuple, Collection
from utils.data_utils import flatten

PREV_LOW = 'Low'
PREV_MEDIUM_LOW = 'Med-Low'
PREV_MEDIUM_HIGH = 'Med-High'
PREV_HIGH = 'High'

pattern = re.compile(r'.+minecore_(?P<resp>.+)-(?P<priv>.+)_')


def class_mapped_files(files: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    responsive, privileged = {}, {}
    for f in files:
        m = pattern.match(f)
        resp = m.group('resp')
        priv = m.group('priv')
        responsive.setdefault(resp, []).append(f)
        privileged.setdefault(priv, []).append(f)
    return responsive, privileged


def rcv1_class_bins(classes: Collection[str], y: npt.NDArray[int]) -> pd.DataFrame:
    df = pd.DataFrame(y.mean(0), columns=['Prev.'], index=classes)
    quartiles = pd.qcut(df['Prev.'], q=4, labels=[PREV_LOW, PREV_MEDIUM_LOW, PREV_MEDIUM_HIGH, PREV_HIGH])
    df['quartiles'] = quartiles
    return df.reset_index().rename(columns={'index': 'Class'})


def get_classes_by_prev(df: pd.DataFrame, prev: str):
    return df[df.quartiles == prev]['Class'].tolist()


def get_average_on_prev(cls_map_dict: Dict[str, str], prev_classes: List[str]) -> pd.DataFrame:
    prev_files = list(flatten(map(cls_map_dict.get, prev_classes)))
    return sum(pd.read_csv(f, sep='\t', index_col=[0, 1]) for f in prev_files) / len(prev_files)


def get_all_prev_dfs(cls_map_dict: Dict[str, str], prev_classes: List[List[str]], rows_to_show: List[str], columns_to_show: List[str]):
    renames = {'AL SLD_priv_cost': 'Post-SLD', 'AL_priv_cost': 'Pre-SLD',
               'MLE_priv_cost': 'Pre-SLD', 'SLD_priv_cost': 'Post-SLD',
               'Rand_priv_cost': 'Pre-SLD', 'Rand SLD_priv_cost': 'Post-SLD'}
    rows_to_show = [] if rows_to_show is None else rows_to_show
    columns_to_show = [] if columns_to_show is None else columns_to_show
    for prev in prev_classes:
        df = get_average_on_prev(cls_map_dict, prev).groupby(level=1).mean().loc[rows_to_show][columns_to_show]
        df = df.rename(index=renames)
        yield df
