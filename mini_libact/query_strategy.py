"""
Code slightly modified from libact https://github.com/ntucllab/libact/
"""
from abc import ABC, abstractmethod
from mini_libact.dataset import Dataset


class QueryStrategy(ABC):

    """Pool-based query strategy
    A QueryStrategy advices on which unlabeled data to be queried next given
    a pool of labeled and unlabeled data.
    """

    def __init__(self, dataset: Dataset, **kwargs):
        self._dataset = dataset
        dataset.on_update(self.update)

    @property
    def dataset(self):
        """The Dataset object that is associated with this QueryStrategy."""
        return self._dataset

    def update(self, entry_id, label):
        """Update the internal states of the QueryStrategy after each queried
        sample being labeled.
        Parameters
        ----------
        entry_id : int
            The index of the newly labeled sample.
        label : float
            The label of the queried sample.
        """
        pass

    def _get_scores(self):
        """Return the score used for making query, the larger the better. Read-only.
        No modification to the internal states.
        Returns
        -------
        (ask_id, scores): list of tuple (int, float)
            The index of the next unlabeled sample to be queried and the score assigned.
        """
        pass

    @abstractmethod
    def make_query(self):
        """Return the index of the sample to be queried and labeled. Read-only.
        No modification to the internal states.
        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.
        """
        pass
