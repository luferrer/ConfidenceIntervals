import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


def evaluate_with_conf_int(labels, samples, metric, conditions=None, num_bootstraps=1000, alpha=5):
    """ Evaluate the metric on the provided data and then run bootstrapping to get a confidence interval.
        - labels: array of labels or any per-sample value needed to compute the metric. 
        - samples: array of decisions/scores/losses for each sample needed to compute the metric.
        - metric: function that takes as arrays of labels and samples, and returns a scalar. 
          If labels is None, the method is called only with samples.
        - conditions: integer array indicating the condition of each sample (in the same order as
          labels and samples)
        - num_bootstraps: number of bootstraps sets to create 
        - alpha: the confidence interval will be computed between alpha/2 and 100-alpha/2 percentiles
    """
    if labels is not None:
        center = metric(labels, samples)
    else:
        center = metric(samples)
    
    bt = Bootstrap(num_bootstraps, metric)
    ci = bt.get_conf_int(samples, labels, conditions, alpha=alpha)
    
    return center, ci



def get_bootstrap_indices(num_samples, conditions=None, random_state=None):
    """ Method that returns the indices for selecting a bootstrap set.
    - num_samples: number of samples in the original set
    - conditions: integer array indicating the condition of each of those samples (in order)
    - random_state: random state for sampling
    If conditions is None, the indices are obtained by sampling an array from 0 to num_samples-1 with 
    replacement. If conditions is not None, the indices are obtained by sampling conditions first
    and then sampling the indices corresponding to the selected conditions. This code is somewhat 
    slow when the number of conditions is large (the slow part is sampling for each condition).
    """

    indices = np.arange(num_samples)
    if conditions is not None:
        # First sample conditions
        unique_conditions = np.unique(conditions)
        bt_conditions = resample(unique_conditions, replace=True, n_samples=len(unique_conditions), 
                                 random_state=random_state)
        
        # Now, for each unique condition selected, sample its indices and repeat them as many times
        # as that condition was selected.
        sel_indices = []
        for s, c in np.c_[np.unique(bt_conditions, return_counts=True)]:
            cond_indices = indices[conditions == s]
            bt_samples_for_cond = resample(cond_indices, replace=True, n_samples=len(cond_indices),
                                           random_state=random_state)   
            sel_indices.append(np.repeat(bt_samples_for_cond, c))
        sel_indices = np.concatenate(sel_indices)
    else:
        sel_indices = resample(indices, replace=True, n_samples=num_samples, random_state=random_state)
        
    return sel_indices


def get_conf_int(values, alpha=5):
        """ Method to obtain the confidence interval from an array of metrics obtained from 
        bootstrapping. Alpha is the level of the test. The confidence interval is computed between 
        alpha/2 and 100-alpha/2 percentiles
        """

        low = np.percentile(values, alpha/2)
        high = np.percentile(values, 100-alpha/2)
        
        return (low, high)



class Bootstrap:

    def __init__(self, num_bootstraps=1000, metric=None):
        """ Class to compute confidence intervals for a metric (e.g. accuracy) using bootstrapping
        - num_bootstraps: number of bootstraps to perform
        - metric: function that takes as input labels and samples and returns a scalar
        """
        self.num_bootstraps = num_bootstraps
        if metric is None:
            self.metric = accuracy_score
        else:
            self.metric = metric

    def get_bootstrap_sets(self, n_samples, conditions=None):
        """ Method to get a list of bootstrap sets. Each set is given by a lists of indices. 
        - n_samples: number of samples in the original set
        - conditions: integer array indicating the condition of each of those samples (in order)
        """
        self.conditions = conditions
        self._indices = []

        for i in range(self.num_bootstraps):
            sel_indices = get_bootstrap_indices(n_samples, self.conditions, random_state=i)
            self._indices.append(sel_indices)

    def get_metric_values_for_bootstrap_sets(self, samples, labels):
        """ Method that computes the metric value for each bootstrap set in self._indices
        - samples: array of decisions/scores/losses for each sample
        - labels: array of labels or any other per-sample information needed to compute the metric 
          This input can be None in which case the metric function is run with samples as the only
          input argument.
        """
        self.samples = samples
        self.labels = labels

        vals = np.zeros(self.num_bootstraps)
        for i, indices in enumerate(self._indices):
            if labels is not None:
                vals[i] = self.metric(self.labels[indices], self.samples[indices])
            else:
                vals[i] = self.metric(self.samples[indices])
        self._scores = vals
        return vals

    def run(self, samples, labels, conditions=None):
        """ Method to compute the confidence interval for the given metric
        - samples: array of decisions for each sample
        - labels: array of labels (0 or 1) for each sample
        - conditions: integer array indicating the condition of each sample (in order)
        """        
        self.get_bootstrap_sets(len(samples), conditions)
        return self.get_metric_values_for_bootstrap_sets(samples, labels)

    
    def get_conf_int(self, samples, labels, conditions=None, alpha=5):
        """ Method to obtain the confidence interval from an array of metrics obtained from bootstrapping
        """
        vals = self.run(samples, labels, conditions)
        self._ci = get_conf_int(vals, alpha)
        return self._ci

        


