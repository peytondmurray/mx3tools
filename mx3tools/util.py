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


# // Gives the forward finite difference coefficients in a slice for a given differentiation order m
# // and number of points n (which determines the order of accuracy). Maximum order of accuracy is always used.
# // Sorry for the bad code, the notation in the original papers is just as bad.
# // Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids",
# // Mathematics of Computation, 51 (184): 699–706
# func fornbergWeights(u float64, x []float64, k int) []float64 {

# 	// if k == 1 && len(x) == 5 {
# 	// 	dx := x[1] - x[0]
# 	// 	C := []float64{3 / (12 * dx), -16 / (12 * dx), 36 / (12 * dx), -48 / (12 * dx), 25 / (12 * dx)}
# 	// 	return C
# 	// }

# 	n := len(x)
# 	C := make([][]float64, k+1)
# 	for i := 0; i < k+1; i++ {
# 		C[i] = make([]float64, n)
# 	}

# 	c1 := float64(1)
# 	c2 := float64(1)
# 	c3 := float64(0)
# 	c4 := x[1] - u
# 	c5 := float64(0)
# 	C[0][0] = 1.0

# 	for i := 0; i < n; i++ {
# 		mn := min(i, k)
# 		c2 = float64(1)
# 		c5 = c4
# 		c4 = x[i] - u

# 		for j := 0; j < i; j++ {
# 			c3 = x[i] - x[j]
# 			c2 *= c3

# 			if j == i-1 {
# 				for s := mn; s > 0; s-- {
# 					C[s][i] = c1 * (float64(s)*C[s-1][i-1] - c5*C[s][i-1]) / c2
# 				}
# 				C[0][i] = -c1 * c5 * C[0][i-1] / c2
# 			}
# 			for s := mn; s > 0; s-- {
# 				C[s][j] = (c4*C[s][j] - float64(s)*C[s-1][j]) / c3
# 			}
# 			C[0][j] = c4 * C[0][j] / c3
# 		}
# 		c1 = c2
# 	}

# 	return C[k]
# }

# @nb.jit(nopython=True)
def fornberg(x, x0, m):

    n = len(x)
    C = np.zeros((n, m+1))

    c1 = 1
    c4 = x[0] - x0

    C[0, 0] = 1
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1
        c5 = c4
        c4 = x[i]-x0

        for j in range(i-1):
            c3 = x[i]-x[j]
            c2 = c2*c3

            if j == i-1:
                for k in range(mn, 1, -1):
                    C[i, k] = c1*(k*C[i-1, k-1] - c5*C[i-1, k])/c2
                C[i, 0] = -c1*c5*C[i-1, 0]/c2
            
            for k in range(mn, 1, -1):
                C[j, k] = (c4*C[j, k] - k*C[j, k-1])/c3
            
            C[j, 0] = c4*C[j, 0]/c3
        c1 = c2

    return C
