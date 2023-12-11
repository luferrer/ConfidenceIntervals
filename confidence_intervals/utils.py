import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def barplot_with_ci(data, figsize=None, colormap='Spectral', outfile=None):
    """ Make a bar plot for the input data. This should be a dictionary with one entry for each
    name in the legend (eg, each system). For each of those, the value should be another dictionary
    with one entry for each group being plotted (eg, each dataset). The entries within that inner 
    dictionary should be a list with the center of the bar (the performance measured on the full 
    test set), and the confidence interval as a list.
    For example:
        
        data = {'sys1': {'db1': (center11, (min11, max11)), 'db2': (center12, (min12, max12))},
                'sys2': {'db1': (center21, (min21, max21)), 'db2': (center22, (min22, max22))}}
        
        """

    cmap = matplotlib.cm.get_cmap(colormap)
    colors = [cmap(i/len(data)) for i in np.arange(len(data))]

    fig, ax = plt.subplots(figsize=figsize)

    # The groups should be the same for all labels
    allgroups = np.unique(np.concatenate([list(lvalues.keys()) for lvalues in data.values()]))

    barWidth = 1/(len(data)+1)

    group_starts = np.arange(len(allgroups))

    for j, (lname, lvalues) in enumerate(data.items()):

        xvalues = group_starts + barWidth * j

        # Plot the bars for the given top label across all groups
        yvalues = [lvalues[group][0] if group in lvalues else 0 for group in allgroups]
        ax.bar(xvalues, yvalues, color=colors[j], width = barWidth, label=lname)

        # Now plot a line on top of the bar to show the confidence interval
        for k, group in enumerate(allgroups):
            ci = lvalues[group][1]
            ax.plot(xvalues[k]*np.ones(2), ci, 'k')

    ax.plot([0,0],[1,1])
    ax.plot([1,1],[1,1])
    ax.set_xticks(group_starts + barWidth * (len(data)-1)/2 , allgroups)
    ax.legend()

    if outfile:
        plt.savefig(outfile)



def create_data(N0, N1, C, random_state=123456, scale=1.0):
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

    loc0 = -1
    loc1 = +1

    for c in condlist:

        shift0 = np.random.normal(0, 0.2, 1)
        shift1 = np.random.normal(0, 0.2, 1)

        idx = (labels == 0) & (conditions == c)
        scores[idx] = np.random.normal(shift0+loc0, scale, np.sum(idx))

        idx = (labels == 1) & (conditions == c)
        scores[idx] = np.random.normal(shift1+loc1, scale, np.sum(idx))

    # Categorical decisions made by thresholding at 0.
    decisions = (scores > 0).astype(int)

    return decisions, labels, conditions


