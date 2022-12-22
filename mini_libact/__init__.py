"""
Installing libact is somehow problematic and not straightforward (especially when not having root access to the system).
Since we only need the QUIRE strategy, we take only the necessary code from the library.
All credits due to libact, https://github.com/ntucllab/libact
"""
from mini_libact.dataset import *
from mini_libact.query_strategy import *
from mini_libact.quire import *
from mini_libact.hierarchical_sampling_strategy import *
from mini_libact.uncertainty_sampling_strategy import *