import numpy as np
from sklearn.metrics import confusion_matrix


class MineCore:

    def __init__(self, pairs: [(str, str)], prior_probabilities: {str: float}, posterior_probabilities: {str: np.ndarray},
                 y_arr: dict, alpha_labels: {(str, str): [float, float]}, ro_r: float,
                 ro_p: float):
        self.pairs = pairs
        self.prior_probabilities = prior_probabilities
        self.posterior_probabilities = posterior_probabilities
        self.y_arr = y_arr
        self.alpha_labels = alpha_labels
        self.ro_r = ro_r
        self.ro_p = ro_p
        self.tau_rs = dict()
        self.tau_ps = dict()
        self.cm_2 = dict()
        self.cm_3 = dict()

    def user_posteriors(self, alpha):
        """
        Returns the Pr^u (posterior probabilities user/us) used in MineCore++
        to model our confidence in the classifier results
        """
        _user_posteriors = dict()
        for label, posteriors in self.posterior_probabilities.items():
            positive_posteriors = posteriors[:, 1] * alpha + (1 - alpha) * self.prior_probabilities[label]
            _user_posteriors[label] = np.array([1 - positive_posteriors, positive_posteriors]).T
        return _user_posteriors

    def run(self, costs):
        """
        Run the three phases of MineCore
        :type costs: cost_structure.Costs
        :returns tau_rs, tau_ps, confusion matrix for phase 2 and phase 3 (cm_2 and cm_3)
        """
        cm_2 = dict()
        cm_3 = dict()
        for c_r, c_p in self.pairs:
            _cm_2, _cm_3 = self.run_on_pair(c_r, c_p, costs)
            cm_2.update(_cm_2)
            cm_3.update(_cm_3)

        return self.tau_rs, self.tau_ps, cm_2, cm_3

    def run_on_pair(self, c_r, c_p, costs):
        cm_2, cm_3 = {}, {}
        cost_matrix = costs.cost_matrix
        cost_r = costs.structure.relev
        cost_p = costs.structure.priv
        manual_labels = costs.manual_labels
        c_r_probs = self.posterior_probabilities[c_r]
        c_p_probs = self.posterior_probabilities[c_p]

        # phase one - automatic step
        risk_h_auto = self.get_h_risk_h(c_r_probs[:, 1], c_p_probs[:, 1], cost_matrix, need_risk=True)[1]

        # phase two - responsiveness
        # cr case
        risk_h_2_cr = self.get_h_risk_h(np.ones_like(c_r_probs[:, 1]), c_p_probs[:, 1], cost_matrix, need_risk=True)[1]

        # ncr case
        risk_h_2_ncr = self.get_h_risk_h(np.zeros_like(c_r_probs[:, 1]), c_p_probs[:, 1], cost_matrix, need_risk=True)[
            1]

        # expected risk
        e_risk_h_2 = risk_h_2_cr * c_r_probs[:, 1] + risk_h_2_ncr * c_r_probs[:, 0]

        delta_or = e_risk_h_2 + np.repeat(cost_r, c_r_probs[:, 0].shape) - risk_h_auto

        self.tau_rs[(c_r, c_p)] = (delta_or < 0).sum()

        c_r_probs_2 = np.copy(c_r_probs[:, 1])

        y_r = self.y_arr[c_r]
        ror_part = round(len(y_r) * self.ro_r)
        y_r[ror_part:] = (y_r[ror_part:] < 0.5).astype(int)  # Transform all 1s to 0s and viceversa
        c_r_probs_2[np.logical_and(delta_or < 0, y_r == 1)] = 1
        c_r_probs_2[np.logical_and(delta_or < 0, y_r == 0)] = 0

        h_2, risk_h_2 = self.get_h_risk_h(c_r_probs_2, c_p_probs[:, 1], cost_matrix, need_risk=True)

        cm_2[(c_r, c_p)] = confusion_matrix(manual_labels[(c_r, c_p)], h_2, labels=[0, 1, 2])

        # phase three - privilege
        # cp case
        risk_h_3_cp = self.get_h_risk_h(c_r_probs_2, np.ones_like(c_p_probs[:, 1]), cost_matrix, need_risk=True)[1]

        # ncp case
        risk_h_3_ncp = self.get_h_risk_h(c_r_probs_2, np.zeros_like(c_p_probs[:, 1]), cost_matrix, need_risk=True)[1]

        # expected risk
        e_risk_h_3 = risk_h_3_cp * c_p_probs[:, 1] + risk_h_3_ncp * c_p_probs[:, 0]

        delta_op = e_risk_h_3 + np.ones(c_p_probs[:, 0].shape) * cost_p - risk_h_2

        self.tau_ps[(c_r, c_p)] = (delta_op < 0).sum()

        c_p_probs_3 = np.copy(c_p_probs[:, 1])

        y_p = self.y_arr[c_p]
        rop_part = round(len(y_p) * self.ro_p)
        y_p[rop_part:] = (y_p[rop_part:] < 0.5).astype(int)  # Transform all 1s to 0s and viceversa
        c_p_probs_3[np.logical_and(delta_op < 0, y_p == 1)] = 1
        c_p_probs_3[np.logical_and(delta_op < 0, y_p == 0)] = 0

        h_3 = self.get_h_risk_h(c_r_probs_2, c_p_probs_3, cost_matrix, need_risk=False)[0]
        cm_3[(c_r, c_p)] = confusion_matrix(manual_labels[(c_r, c_p)], h_3, labels=[0, 1, 2])
        return cm_2, cm_3

    def run_plusplus(self, costs):
        """
        Run the three phases of MineCore Plus Plus (ie. uses alpha values)
        :type costs: cost_structure.Costs
        :returns tau_rs, tau_ps, confusion matrix for phase 2 and phase 3 (cm_2 and cm_3)
        """
        cost_matrix = costs.cost_matrix
        cost_r = costs.structure.relev
        cost_p = costs.structure.priv
        manual_labels = costs.manual_labels
        cm_2 = dict()
        cm_3 = dict()
        for c_r, c_p in self.pairs:
            c_r_probs = self.posterior_probabilities[c_r]
            c_p_probs = self.posterior_probabilities[c_p]

            alpha_cr, alpha_cp = self.alpha_labels[(c_r, c_p)]
            user_posteriors_cr = self.user_posteriors(alpha_cr)[c_r]
            user_posteriors_cp = self.user_posteriors(alpha_cp)[c_p]

            # phase one - automatic step
            risk_h_auto = self.get_h_risk_h(user_posteriors_cr[:, 1], user_posteriors_cp[:, 1], cost_matrix, need_risk=True)[1]

            # phase two - responsiveness
            # cr case
            risk_h_2_cr = self.get_h_risk_h(np.ones_like(c_r_probs[:, 1]) * self.ro_r, c_p_probs[:, 1], cost_matrix, need_risk=True)[1]

            # ncr case
            risk_h_2_ncr = self.get_h_risk_h(np.ones_like(c_r_probs[:, 1]) - self.ro_r, c_p_probs[:, 1], cost_matrix, need_risk=True)[1]

            # expected risk
            e_risk_h_2 = risk_h_2_cr * user_posteriors_cr[:, 1] + risk_h_2_ncr * user_posteriors_cr[:, 0]

            delta_or = e_risk_h_2 + np.repeat(cost_r, user_posteriors_cr[:, 0].shape) - risk_h_auto

            self.tau_rs[(c_r, c_p)] = (delta_or < 0).sum()

            c_r_probs_2 = np.copy(user_posteriors_cr[:, 1])

            y_r = self.y_arr[c_r]
            ror_part = round(len(y_r) * self.ro_r)
            y_r[:ror_part] = y_r[:ror_part]
            y_r[ror_part:] = (y_r[ror_part:] < 0.5).astype(int)  # Transform all 1s to 0s and viceversa
            c_r_probs_2[np.logical_and(delta_or < 0, y_r == 1)] = 1 * self.ro_r
            c_r_probs_2[np.logical_and(delta_or < 0, y_r == 0)] = 1 - self.ro_r

            h_2, risk_h_2 = self.get_h_risk_h(c_r_probs_2, user_posteriors_cp[:, 1], cost_matrix, need_risk=True)

            cm_2[(c_r, c_p)] = confusion_matrix(manual_labels[(c_r, c_p)], h_2, labels=[0, 1, 2])

            # phase three - privilege
            # cp case
            risk_h_3_cp = self.get_h_risk_h(c_r_probs_2, np.ones_like(c_p_probs[:, 1]) * self.ro_p, cost_matrix, need_risk=True)[1]

            # ncp case
            risk_h_3_ncp = self.get_h_risk_h(c_r_probs_2, np.ones_like(c_p_probs[:, 1]) - self.ro_p, cost_matrix, need_risk=True)[1]

            # expected risk
            e_risk_h_3 = risk_h_3_cp * user_posteriors_cp[:, 1] + risk_h_3_ncp * user_posteriors_cp[:, 0]

            delta_op = e_risk_h_3 + np.ones(c_p_probs[:, 0].shape) * cost_p - risk_h_2

            self.tau_ps[(c_r, c_p)] = (delta_op < 0).sum()

            c_p_probs_3 = np.copy(c_p_probs[:, 1])

            y_p = self.y_arr[c_p]
            rop_part = round(len(y_p) * self.ro_p)
            y_p[:rop_part] = y_p[:rop_part]
            y_p[rop_part:] = (y_p[rop_part:] < 0.5).astype(int)  # Transform all 1s to 0s and viceversa
            c_p_probs_3[np.logical_and(delta_op < 0, y_p == 1)] = 1 * self.ro_p
            c_p_probs_3[np.logical_and(delta_op < 0, y_p == 0)] = 1 - self.ro_p

            h_3 = self.get_h_risk_h(c_r_probs_2, c_p_probs_3, cost_matrix, need_risk=False)[0]
            cm_3[(c_r, c_p)] = confusion_matrix(manual_labels[(c_r, c_p)], h_3, labels=[0, 1, 2])

        return self.tau_rs, self.tau_ps, cm_2, cm_3

    @staticmethod
    def get_h_risk_h(c_r_prob, c_p_prob, cost_matrix, need_risk=True):
        c_P_prob = c_r_prob * (1 - c_p_prob)
        c_L_prob = c_r_prob * c_p_prob
        c_W_prob = 1 - c_r_prob

        risk_P = cost_matrix['P']['P'] * c_P_prob + cost_matrix['P']['L'] * c_L_prob + cost_matrix['P']['W'] * c_W_prob
        risk_L = cost_matrix['L']['P'] * c_P_prob + cost_matrix['L']['L'] * c_L_prob + cost_matrix['L']['W'] * c_W_prob
        risk_W = cost_matrix['W']['P'] * c_P_prob + cost_matrix['W']['L'] * c_L_prob + cost_matrix['W']['W'] * c_W_prob

        # NOTE: the order of the following vstack must follow the label_to_idx mapping
        risks = np.vstack((risk_P, risk_L, risk_W)).T
        h = risks.argmin(1)
        if need_risk:
            risk_h = np.asarray([risks[i, h[i]] for i in range(h.shape[0])])
        else:
            risk_h = None
        return h, risk_h

    def get_beta_i(self, i: str, cost_matrix, c_r_priors, c_p_priors, alpha):
        c_P_prob = c_r_priors * (1 - c_p_priors)
        c_L_prob = c_r_priors * c_p_priors
        c_W_prob = 1 - c_r_priors

        return (1 - alpha) * \
               (cost_matrix[i]['P'] * c_P_prob + cost_matrix[i]['L'] * c_L_prob + cost_matrix[i]['W'] * c_W_prob)
