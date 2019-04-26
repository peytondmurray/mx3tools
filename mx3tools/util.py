import numpy as np


def validate_pdf(bins, hist, tol=0.01):
    """Given a set of bins and a histogram, check whether the area adds to 1.

    Parameters
    ----------
    bins : np.ndarray
        Bin edges, including the rightmost edge; len(bins) = len(hist) + 1
    hist : np.ndarray
        hist[i] gives the number of events between bins[i] and bins[i+1]
    tol : float
        Tolerance used for checking whether the area is close to 1. By default this is set to 1%, which seems to
        be a good place to start, allowing for numerical errors arising from the rectangular integration.
    """

    area = np.sum(np.diff(bins)*hist)

    if not np.isclose(area, 1, rtol=tol, atol=0):
        raise ValueError(f'Histogram isnot normalized correctly. Area = {area}')


def dict_add(d1, d2):
    """Add two dicts together without overwriting any values in the first dict. Returns a new dict without modifying
    the input dicts.

    Parameters
    ----------
    d1 : dict
        Main dictionary
    d2 : dict
        Dictionary of values to try to add

    Returns
    -------
    dict
        Combined dictionary, containing the elements of d1 and d2, except in the case where a key in d2 was the same
        as in d1, in which case only the value in d1 is used.
    """
    d = d1.copy()
    d.update({k: v for k, v in d2.items() if k not in d1})
    return d
