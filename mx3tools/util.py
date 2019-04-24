import numpy as np


def validate_pdf(bins, hist):
    """Given a set of bins and a histogram, check whether the area adds to 1.

    Parameters
    ----------
    bins : np.ndarray
        Bin edges, including the rightmost edge; len(bins) = len(hist) + 1
    hist : np.ndarray
        hist[i] gives the number of events between bins[i] and bins[i+1]
    """

    return np.isclose(np.sum(np.diff(bins)*hist), 1)
