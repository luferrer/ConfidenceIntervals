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


def get_conf_int(values, alpha=5):
        """ Method to obtain the confidence interval from an array of metrics obtained from bootstrapping
        """

        low = np.percentile(values, alpha/2)
        high = np.percentile(values, 100-alpha/2)
        
        return (low, high)


def evaluate_metric_with_conf_int(y_true, y_pred, metric, conditions=None, num_bootstraps=1000, alpha=5):
    """ Evaluate the metric on the provided data and then run bootstrapping to get a confidence interval.
        - y_true: array of labels or any per-sample value needed to compute the metric
        - y_pred: array of decisions/scores/losses for each sample needed to compute the metric
        - metric: function that takes as input y_true and y_pred (or sampled versions of those inputs), 
          and returns a scalar
        - conditions: integer array indicating the condition of each sample (in the same order as
          y_true and y_pred)
        - num_bootstraps: number of bootstraps sets to create 
        - alpha: confidence interval will be computed between alpha/2 and 100-alpha/2 percentiles
    """

    center = metric(y_true, y_pred)
    
    bt = Bootstrap(num_bootstraps, metric)
    ci = bt.get_conf_int(y_pred, y_true, conditions, alpha=5)
    
    return center, ci

class Bootstrap:

    def __init__(self, num_bootstraps=1000, metric=None):
        """ Class to compute confidence intervals for a metric (e.g. accuracy) using bootstrapping
        - y_pred: array of decisions/scores/losses for each sample in the test dataset
        - y_true: array of labels or any additional value about each sample needed to compute the metric
        - conditions: integer array indicating the condition of each sample (in order)
        - num_bootstraps: number of bootstraps to perform
        - metric: function that takes as input y_true and y_pred (or sampled versions of those inputs), 
          and returns a scalar
        """

        self.num_bootstraps = num_bootstraps
        if metric is None:
            self.metric = accuracy_score
        else:
            self.metric = metric

    def get_list_of_bootstrap_indices(self, n_samples, conditions=None):
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

    def get_metric_values_for_bootstrap_sets(self, y_pred, y_true):
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

    def run(self, y_pred, y_true, conditions=None):
        """ Method to compute the confidence interval for the given metric
        - y_pred: array of decisions for each sample
        - y_true: array of labels (0 or 1) for each sample
        - conditions: integer array indicating the condition of each sample (in order)
        """        
        self.get_list_of_bootstrap_indices(len(y_pred), conditions)
        return self.get_metric_values_for_bootstrap_sets(y_pred, y_true)

    
    def get_conf_int(self, y_pred, y_true, conditions=None, alpha=5):
        """ Method to obtain the confidence interval from an array of metrics obtained from bootstrapping
        """
        vals = self.run(y_pred, y_true, conditions)
        self._ci = get_conf_int(vals, alpha)
        return self._ci

        


