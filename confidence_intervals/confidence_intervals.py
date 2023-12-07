import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


def get_bootstrap_indices(N=None, conditions=None, random_state=None):
    """ Method that returns the indices for selecting a bootstrap set.
    - num_samples: number of samples in the original set
    - conditions: integer array indicating the condition of each of those samples (in order)
    If conditions is None, the indices are obtained by sampling an array from 0 to N-1 with 
    replacement. If conditions is not None, the indices are obtained by sampling conditions first
    and then retrieving the sample indices corresponding to the selected conditions.
    """

    indices = np.arange(N)
    if conditions is not None:
        unique_conditions = np.unique(conditions)
        bt_conditions = resample(unique_conditions, replace=True, n_samples=len(
            unique_conditions), random_state=random_state)
        sel_indices = np.concatenate(
            [indices[np.where(conditions == s)[0]] for s in bt_conditions])
    else:
        sel_indices = resample(indices, replace=True,
                               n_samples=N, random_state=random_state)
    return sel_indices
    
class Bootstrap:

    def __init__(self, num_bootstraps=1000, metric=None, alpha=5):
        """ Class to compute confidence intervals for a metric (e.g. accuracy) using bootstrapping
        - y_pred: array of decisions for each sample
        - y_true: array of labels (0 or 1) for each sample
        - conditions: integer array indicating the condition of each sample (in order)
        - num_bootstraps: number of bootstraps to perform
        - metric: function that takes as input decisions, labels, and conditions and returns a scalar
        """

        self.num_bootstraps = num_bootstraps
        if metric is None:
            self.metric = accuracy_score
        else:
            self.metric = metric

        self.alpha = alpha

    def fit(self, n_samples, conditions=None):
        """ Method to compute the confidence interval for the given metric
        - n_samples: number of samples in the original set
        - conditions: integer array indicating the condition of each of those samples (in order)
        """
        self.conditions = conditions
        self._indices = []

        for i in range(self.num_bootstraps):
            sel_indices = get_bootstrap_indices(
                n_samples, self.conditions, random_state=i)
            self._indices.append(sel_indices)

    def transform(self, y_pred, y_true):
        """ Method to compute the confidence interval for the given metric
        - y_pred: array of decisions for each sample
        - y_true: array of labels (0 or 1) for each sample
        """
        self.y_pred = y_pred
        self.y_true = y_true

        vals = np.zeros(self.num_bootstraps)
        for i, indices in enumerate(self._indices):
            vals[i] = self.metric(self.y_pred[indices], self.y_true[indices])
        self._scores = vals
        return vals

    def fit_transform(self, y_pred, y_true, conditions=None):
        """ Method to compute the confidence interval for the given metric
        - y_pred: array of decisions for each sample
        - y_true: array of labels (0 or 1) for each sample
        - conditions: integer array indicating the condition of each sample (in order)
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.conditions = conditions
        self._indices = []

        vals = np.zeros(self.num_bootstraps)
        for i in range(self.num_bootstraps):
            sel_indices = get_bootstrap_indices(
                len(self.y_pred), self.conditions, random_state=i)
            self._indices.append(sel_indices)
            vals[i] = self.metric(
                self.y_pred[sel_indices], self.y_true[sel_indices])
        self._scores = vals
        return vals

    def get_conf_int(self, alpha=None, print_result=False):
        """ Method to obtain the confidence interval from an array of metrics obtained from bootstrapping
        """
        if alpha is None:
            alpha = self.alpha

        low = np.percentile(self._scores, alpha/2)
        high = np.percentile(self._scores, 100-alpha/2)
        self._ci = (low, high)
        if print_result:
            print(f"Confidence interval: {low:5.2f}  {high:5.2f}")

        return low, high


