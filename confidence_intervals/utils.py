

import numpy as np


def get_conf_int(vals, alpha=5):
    """ Method to obtain the confidence interval from an array of metrics obtained from bootstrapping
    """
    print("Confidence interval: %5.2f  %5.2f" %
          (np.percentile(vals, alpha/2), np.percentile(vals, 100-alpha/2)))


def create_data(N0, N1, C, random_state=123456):
    """ Create a toy dataset for binary classification with N0 samples from class 0, N1 
     samples from class 1, and C conditions. 
    """

    N = N0 + N1

    # Assign toy conditions to the samples which play the role of some correlation-inducing factor
    # Assume there are C distinct conditions
    condlist = np.arange(C)
    np.random.seed(random_state)
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
    decisions = (scores > 0).astype(int)

    return decisions, labels, conditions
