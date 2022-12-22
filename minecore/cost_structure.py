import numpy as np
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from minecore import MineCore


CostStructure = namedtuple("CostStructure", ["relev", "priv", "PP", "PL", "PW", "LP", "LL", "LW", "WP", "WL", "WW"])

cost_structure_1 = CostStructure(1., 5., 0., 600., 5., 150., 0., 3., 15., 15., 0)
cost_structure_2 = CostStructure(1., 5., 0., 100., 0.03, 10., 0., 2., 8., 8., 0.)
cost_structure_3 = CostStructure(1., 5., 0., 1000., 0.1, 1., 0., 1., 1., 1., 0.)


class Costs:

    def __init__(self, structure: CostStructure, pairs: list, posterior_probabilities: {str: float}, y_arr: dict, alphas=None,
                 prior_probabilities=None, ro_r=1, ro_p=1):
        self.structure = structure
        self.pairs = pairs
        self.posterior_probabilities = posterior_probabilities
        self.prior_probabilities = prior_probabilities
        self.y_arr = y_arr
        self.alphas = alphas
        self.ro_r = ro_r
        self.ro_p = ro_p
        self._cost_matrix = None
        self._label_to_idx = None
        self._cm_auto = dict()
        self._manual_labels = dict()

    @property
    def cost_matrix(self) -> {str: {str: float}}:
        if self._cost_matrix is None:
            self._cost_matrix = {
                'P': {'P': self.structure.PP, 'L': self.structure.PL, 'W': self.structure.PW},
                'L': {'P': self.structure.LP, 'L': self.structure.LL, 'W': self.structure.LW},
                'W': {'P': self.structure.WP, 'L': self.structure.WL, 'W': self.structure.WW},
            }
        return self._cost_matrix

    @property
    def cost_matrix_array(self) -> np.array:
        return np.array([
            [self.structure.PP, self.structure.PL, self.structure.PW],
            [self.structure.LP, self.structure.LL, self.structure.LW],
            [self.structure.WP, self.structure.WL, self.structure.WW]
        ])

    @property
    def label_to_idx(self) -> {str: int}:
        if self._label_to_idx is None:
            self._label_to_idx = {'P': 0, 'L': 1, 'W': 2}
        return self._label_to_idx

    @property
    def manual_labels(self) -> {(str, str): float}:
        if not self._manual_labels:
            for c_r, c_p in self.pairs:
                y_r = self.y_arr[c_r]
                y_p = self.y_arr[c_p]
                y = np.zeros(y_p.shape, dtype=int)
                y[y_r == 0] = self.label_to_idx['W']
                y[np.logical_and(y_r == 1, y_p == 1)] = self.label_to_idx['L']
                y[np.logical_and(y_r == 1, y_p == 0)] = self.label_to_idx['P']
                self._manual_labels[(c_r, c_p)] = y
        return self._manual_labels

    @property
    def cm_auto(self) -> {(str, str): np.array}:
        if not self._cm_auto:
            minecore = MineCore(self.pairs, self.prior_probabilities, self.posterior_probabilities,
                                self.y_arr, self.alphas, 1, 1)
            for c_r, c_p in self.pairs:
                if self.alphas is None:
                    c_r_probs = self.posterior_probabilities[c_r]
                    c_p_probs = self.posterior_probabilities[c_p]
                else:
                    alpha_cr, alpha_cp = self.alphas[(c_r, c_p)]
                    c_r_probs = minecore.user_posteriors(alpha_cr)[c_r]
                    c_p_probs = minecore.user_posteriors(alpha_cp)[c_p]

                h = MineCore.get_h_risk_h(c_r_probs[:, 1], c_p_probs[:, 1], self.cost_matrix, need_risk=False)[0]

                self._cm_auto[(c_r, c_p)] = confusion_matrix(self.manual_labels[(c_r, c_p)], h, labels=[0, 1, 2])
        return self._cm_auto

    def get_manual_costs(self, n_docs: int) -> ({(str, str): float},):
        cost_manual = dict()
        cost_manual_ann = dict()
        cost_manual_misc = dict()
        for c_r, c_p in self.pairs:
            matrix_r, matrix_p = self.__manual_contingency_table(c_r, c_p)
            cost_ann = self.structure.relev * n_docs + self.structure.priv * (matrix_r[0][0] + matrix_r[0][1])
            cost_misc = self.__manual_misclass_matrix(matrix_r, matrix_p, n_docs)
            cost_manual_ann[(c_r, c_p)] = cost_ann
            cost_manual_misc[(c_r, c_p)] = cost_misc
            cost_manual[(c_r, c_p)] = cost_ann + cost_misc
        return cost_manual, cost_manual_ann, cost_manual_misc

    def __manual_misclass_matrix(self, matrix_r, matrix_p, n_docs):
        # normalize matrices
        matrix_r = matrix_r / matrix_r.sum()
        matrix_p = matrix_p / matrix_p.sum()

        # compute probabilities of events
        pp = matrix_r[0][0] * matrix_p[1][1]  # TPr * TNp
        pl = matrix_r[0][0] * matrix_p[1][0]  # TPr * FNp
        pw = (matrix_r[0][1] * matrix_p[1][0]) + (matrix_r[0][1] * matrix_p[1][1])  # FPr * FNp + FPr * TNp
        lp = matrix_r[0][0] * matrix_p[0][1]  # TPr * FPp
        ll = matrix_r[0][0] * matrix_p[0][0]  # TPr * TPp
        lw = (matrix_r[0][1] * matrix_p[0][1]) + (matrix_r[0][1] * matrix_p[0][0])  # FPr * FPp + FPr * TPp
        wp = (matrix_r[1][0] * matrix_p[1][0]) + (matrix_r[1][0] * matrix_p[1][1])  # FNr * FNp + FNr * TNp
        wl = (matrix_r[1][0] * matrix_p[0][1]) + (matrix_r[1][0] * matrix_p[0][0])  # FNr * FPp + FNr * TPp
        ww = (matrix_r[1][1] * matrix_p[0][0]) + (matrix_r[1][1] * matrix_p[0][1]) + (matrix_r[1][1] * matrix_p[1][0]) + \
             (matrix_r[1][1] * matrix_p[1][1])  # TNr * TPp + TNr * FPp + TNr * FNp + TNr * TNp

        """
        [PP, PL, PW],
        [LP, LL, LW],
        [WP, WL, WW]
        """
        contingency_matrix = np.array([
            [pp * n_docs, pl * n_docs, pw * n_docs],
            [lp * n_docs, ll * n_docs, lw * n_docs],
            [wp * n_docs, wl * n_docs, ww * n_docs]
        ])

        return (contingency_matrix * self.cost_matrix_array).sum()

    def __manual_contingency_table(self, c_r, c_p):
        y_r = self.y_arr[c_r]
        y_p = self.y_arr[c_p]
        n_pos_r = y_r[y_r == 1].shape[0]
        n_neg_r = y_r[y_r == 0].shape[0]
        n_pos_p = y_p[y_p == 1].shape[0]
        n_neg_p = y_p[y_p == 0].shape[0]
        matrix_r = np.array([
            [self.ro_r * n_pos_r, (1 - self.ro_r) * n_neg_r],
            [(1 - self.ro_r) * n_pos_r, self.ro_r * n_neg_r]]
        )
        matrix_p = np.array([
            [self.ro_p * n_pos_p, (1 - self.ro_p) * n_neg_p],
            [(1- self.ro_p) * n_pos_p, self.ro_p * n_neg_p]
        ])

        return matrix_r, matrix_p

    def get_automatic_costs(self) -> ({(str, str): float},):
        return self.__compute_costs(
            self.cm_auto,
            lambda _, _i: 0.0
        )

    def get_second_phase_costs(self, cm_2, tau_rs) -> ({(str, str): float},):
        return self.__compute_costs(
            cm_2,
            lambda c_r, c_p: tau_rs[(c_r, c_p)] * self.structure.relev
        )

    def get_third_phase_costs(self, cm_3, tau_rs, tau_ps) -> ({(str, str): float},):
        return self.__compute_costs(
            cm_3,
            lambda c_r, c_p: tau_rs[(c_r, c_p)] * self.structure.relev + tau_ps[(c_r, c_p)] * self.structure.priv
        )

    def __compute_costs(self, cm_dict: dict, compute_annotation_func):
        overall_costs = dict()
        annotation_costs = dict()
        misclas_costs = dict()
        for c_r, c_p in self.pairs:
            cm = cm_dict[(c_r, c_p)]
            cost_misc = 0.0
            for true_label in self.label_to_idx.keys():
                for pred_label in self.label_to_idx.keys():
                    cost_misc += self.cost_matrix[true_label][pred_label] * cm[
                        self.label_to_idx[pred_label], self.label_to_idx[true_label]]
            cost_ann = compute_annotation_func(c_r, c_p)
            annotation_costs[(c_r, c_p)] = cost_ann
            misclas_costs[(c_r, c_p)] = cost_misc
            overall_costs[(c_r, c_p)] = cost_ann + cost_misc
        return overall_costs, annotation_costs, misclas_costs
