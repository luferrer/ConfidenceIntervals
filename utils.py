import numpy as np
from sklearn.utils import resample


def get_bootstrap_indices(N, conditions=None):
    """ Method that returns the indices for selecting a bootstrap set.
    - num_samples: number of samples in the original set
    - conditions: integer array indicating the condition of each of those samples (in order)
    If conditions is None, the indices are obtained by sampling an array from 0 to N-1 with 
    replacement. If conditions is not None, the indices are obtained by sampling conditions first
    and then retrieving the sample indices corresponding to the selected conditions.
    """

    indices = np.arange(N)

    if conditions is not None:
        if len(conditions) != N:
            raise Exception("The number of conditions should be equal to N, the first argument")
        unique_conditions = np.unique(conditions)
        bt_conditions = resample(unique_conditions, replace=True, n_samples=len(unique_conditions))
        sel_indices = np.concatenate([indices[np.where(conditions == s)[0]] for s in bt_conditions])
    else:
        sel_indices = resample(indices, replace=True, n_samples=N)
        
    return sel_indices


def get_conf_int(vals, alpha=5):
    """ Method to obtain the confidence interval from an array of metrics obtained from bootstrapping
    """
    print("Confidence interval: %5.2f  %5.2f"%(np.percentile(vals, alpha/2), np.percentile(vals, 100-alpha/2)))
 

def create_data(N0, N1, C):
    """ Create a toy dataset for binary classification with N0 samples from class 0, N1 
     samples from class 1, and C conditions. 
    """
    
    N = N0 + N1

    # Assign toy conditions to the samples which play the role of some correlation-inducing factor 
    # Assume there are C distinct conditions
    C = 10
    condlist = np.arange(C)
    conditions = np.random.choice(C, N)

    # Generate scores with Gaussian distribution centered at -1 for one class and at 1 for 
    # the other class. For each condition, assume the scores are slightly shifted with respect to 
    # those centers.
    scores = np.zeros(N)
    labels = np.r_[np.zeros(N0), np.ones(N1)]

    scale = 1.0
    loc0 = -1
    loc1 = +1

    for c in condlist:

        shift0 = np.random.normal(0, 0.3, 1)
        shift1 = np.random.normal(0, 0.3, 1)

        idx = (labels == 0) & (conditions == c)
        scores[idx] = np.random.normal(shift0+loc0, scale, np.sum(idx))

        idx = (labels == 1) & (conditions == c)
        scores[idx] = np.random.normal(shift1+loc1, scale, np.sum(idx))

    # Categorical decisions made by thresholding at 0. 
    decisions = (scores>0).astype(int)

    return decisions, labels, conditions
