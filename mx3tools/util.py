import numpy as np
import numba as nb


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


# Gives the forward finite difference coefficients in a slice for a given differentiation order m
# and number of points n (which determines the order of accuracy). Maximum order of accuracy is always used.
# Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids",
# Mathematics of Computation, 51 (184): 699–706
# @nb.jit(nopython=True)
def fornberg(x, x0, m):
    N = len(x)
    if m >= N:
        # Number of points given must be greater than the order of the derivative
        return np.nan

    c1 = 1
    c4 = x[0]-x0
    C = np.zeros((N, m+1))
    C[0, 0] = 1
    for n in range(1, N):
        mn = np.arange(0, min(n, m) + 1)
        c2, c5, c4 = 1, c4, x[n] - x0
        for v in range(n):
            c3 = x[n] - x[v]
            c2, c6, c7 = c2 * c3, mn * C[v, mn-1], C[v, mn]
            C[v, mn] = (c4 * c7 - c6) / c3
        C[n, mn] = c1 * (c6 - c5 * c7) / c2
        c1 = c2
    return C[:, -1]

# @nb.jit(nopython=True)
def diff(x, y, m=1, n=1):
    dydx = np.zeros(len(y))

    for i in range(n):
        print(i, end='\r')
        dydx[i] = np.sum(fornberg(x[:n], x[i], m)*x[:n])

    for i in range(n, len(x)-n):
        print(i, end='\r')
        dydx[i] = np.sum(fornberg(x[i-n:i+n+1], x[i], m)*x[i-n:i+n+1])

    for i in range(len(x)-m, len(x)):
        print(i, end='\r')
        dydx[i] = np.sum(fornberg(x[-n:], x[i], m)*x[-n:])

    return dydx
