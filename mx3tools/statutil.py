# Methods related to statistical analysis of simulation output files

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import numba as nb
import pathlib
import warnings
import time

from . import datautil
from . import util


class Seismograph:
    """Finds the avalanches and corresponding sizes and durations in a signal.

    Parameters
    ----------
    t : np.ndarray
        Time
    v : np.ndarray
        Signal to be watched for avalanches
    vt : float
        Threshold value; events are found when the signal crosses this threshold
    s : np.ndarray
        Signal to be integrated during an avalanche to get avalanche sizes; by default, this is v-vt.
        But it can also be different, e.g., a domain wall avalanche can be thresholded on the velocity,
        but can be integrated to get the avalanche sizes in terms of Axy and Az.

        If this parameter is applied, no threshold is subtracted from s before it is integrated.

    """

    def __init__(self, t, v, vt, s=None):

        self.t = t
        self.v = v
        self.vt = vt
        self.s = s if s is not None else v - vt

        if t.shape != v.shape != s.shape:
            warnings.warn(f't, v, and s must be the same shape. (t, v, s): ({len(t)}, {len(v)}, {len(s)})')
            self.sizes = np.zeros(0)
            self.durations = np.zeros(0)
            return

        self.istart, self.istop = _events(v, vt)
        # self.istart, self.istop = _remove_length1_events(*_events(v, vt))
        self.tstart, self.tstop = self.t[self.istart], self.t[self.istop]
        self.durations = self.tstop - self.tstart

        self.sizes = _event_sizes(self.t, self.s, self.istart, self.istop)

        return


def _remove_length1_events(istart, istop):
    longer_than_1 = istop - istart > 1
    return istart[longer_than_1], istop[longer_than_1]


def _start_indices(v, vt):
    """Find the starting indices of each event in the signal.

    Parameters
    ----------
    v : np.ndarray
        Signal to be searched for avalanches
    vt : float
        Threshold value; events are found when the signal crosses this threshold

    Returns
    -------
    np.ndarray
        The starting indices of each event
    """

    return np.nonzero(np.logical_and(v[1:] > vt, v[:-1] <= vt))[0]+1


def _end_indices(v, vt):
    """Find the stopping indices of each event in the signal.

    Parameters
    ----------
    v : np.ndarray
        Signal to be searched for avalanches
    vt : float
        Threshold value; events are found when the signal crosses this threshold

    Returns
    -------
    np.ndarray
        The ending indices of each event
    """

    return np.nonzero(np.logical_and(v[1:] <= vt, v[:-1] > vt))[0]+1


def _events(v, vt):
    """Return the starting and stopping indices of each event in the signal. The returned arrays have same size. The
    first event is always ignored. If v[-1] > vt, the last event is considered to end at v[-1], unless v[-2] < vt,
    in which case the last event is dropped to avoid issues with finding the event size.

    Parameters
    ----------
    v : np.ndarray
        Signal to be searched for avalanches
    vt : float
        Threshold value; events are found when the signal crosses this threshold

    Returns
    -------
    (np.ndarray, np.ndarray)
        Pair of arrays: first array contains the starting indices of each event, the second array contains the ending
        indices of each event
    """

    i_start = _start_indices(v, vt)
    i_stop = _end_indices(v, vt)

    if i_start[0] > i_stop[0]:
        i_stop = i_stop[1:]
    if i_start[-1] > i_stop[-1]:
        i_start = i_start[:-1]

    if i_start.shape != i_stop.shape:
        raise ValueError('Starting and stopping indices of avalanches do not have same number of elements.')

    return i_start, i_stop


def _event_sizes(t, s, i_start, i_stop):
    """Compute the size of each avalanche in the signal. The size of an avalanche is the integral of the signal over
    the time the signal is above the threshold. This integration is done using rectangles, with

        dt[i] = (t[i+1]-t[i-1])/2

    except at the edges, where the forward (for t[0]) or backward (for t[-1]) differences are taken.

    Parameters
    ----------
    t : np.ndarray
        Time
    s : np.ndarray
        Signal to be integrated
    i_start : np.ndarray
        Array of starting indices of the events
    i_stop : np.ndarray
        Array of ending indices of the events

    Returns
    -------
    np.ndarray
        Size of each event
    """

    # Calculate the central difference, and use forward and backward differences for the edge to preserve length
    dt = np.hstack((np.array([t[1] - t[0]]), (t[2:]-t[:-2])*0.5, np.array([t[-1]-t[-2]])))

    ret = np.empty(i_start.shape[0])
    for i in range(len(i_start)):
        ret[i] = np.sum(s[i_start[i]:i_stop[i]]*dt[i_start[i]:i_stop[i]], axis=0)

    return ret


def bin_avg_event_shape(data, duration=None, tol=None, drange=None, nbins=None, norm=True, normy=False):
    """Get the binned-average event shapes from the data.  Each event has a time array (t) and signal array (s).

    If tol is specified and range is None, events of duration d which fall within

        duration - tol < d < duration + tol

    are collected. If drange is specified, events of duration d which fall within

        drange[0] < d < drange[1]

    are collected. The time arrays are then normalized to the interval [0, 1]. The time-axis is then divided into nbins
    number of time bins, and the value of s in each bin is averaged across all events.

    Parameters
    ----------
    data : datautil.SimRun or datautil.Simdata
        Data to analyze
    duration : float
        Set the duration of the bins to average. If None, drange is used.
    tol : float or None
        Sets the tolerance determining which events to include in the average. If None, drange is used.
    drange : (float, float) or None
        Sets the range determining which events to include in the average. If None, tol is used.
    nbins : int
        Number of bins to divide the time axis into. If nbins==None, uses the smallest number of bins possible; see
        docstring for bin_avg()
    norm : bool
        Set to True to normalize the time to the interval [0, 1]

    Returns
    -------
    4-tuple of np.ndarray
        (event times, event signals, binned-normalized time, binned-average signal)
    """

    if drange is None and tol is not None and duration is not None:
        t, s = data.events_by_duration(duration-tol, duration+tol)
    elif tol is None and duration is None and drange is not None:
        t, s = data.events_by_duration(drange[0], drange[1])
    else:
        raise ValueError('Must specify either a range or tolerance for bin_avg_event_shape()')

    t = normalize_t(t)
    tbin, sbin = bin_avg(t, s, nbins=nbins, norm=norm, normy=normy)

    return t, s, tbin, sbin


def bin_avg(t, s, nbins=None, norm=True, normy=False):
    """Bin and average the input signals. The times of each event are normalized from 0 to 1 if norm=True.

    Parameters
    ----------
    t : list or np.ndarray
        Can be
        1. A list of np.ndarrays, each containing a set of times (usually corresponding to measurements durng
           an avalanche)
        2. A 1D np.ndarray containing times
    s : list or np.ndarray
        Values of the signal measured at times t. Must be same shape as t.
    nbins : int or None
        Number of bins to use. If None, then the number of bins is set equal to the length of the smallest event array.
    norm : bool
        Scale the t-axis to [0, 1].

    Returns
    -------
    t_bin : np.ndarray
        Binned timesteps
    s_bin : np.ndarray
        Average value of the signal at each timestep
    """

    if norm:
        t = normalize_t(t)

    if isinstance(t, list):
        f_t = np.hstack(t).flatten()
    else:
        f_t = t.flatten()

    if isinstance(s, list):
        f_s = np.hstack(s).flatten()
    else:
        f_s = s.flatten()

    if nbins is None:

        nbins = np.min([len(_t) for _t in t])
        t_bin = np.linspace(np.min(f_t), np.max(f_t), nbins+1)  # Array of bin edges
        s_bin = np.zeros(t_bin.shape)
    else:
        t_bin = np.linspace(np.min(f_t), np.max(f_t), nbins+1)
        s_bin = np.zeros(nbins+1)

    for i in range(nbins):
        in_bin_i = np.nonzero(np.logical_and(t_bin[i] <= f_t, f_t < t_bin[i+1]))
        s_bin[i] = np.mean(f_s[in_bin_i])

    # in_last_bin = np.nonzero(np.logical_and(t_bin[-2] <= f_t, f_t <= t_bin[-1]))
    # s_bin[-1] = np.mean(f_s[in_last_bin])

    if normy == 'max':
        return t_bin, (s_bin - np.min(s_bin))/(np.max(s_bin)-np.min(s_bin))
    elif normy == 'area':
        return t_bin, (s_bin)/np.trapz(s_bin, t_bin)
    else:
        return t_bin, s_bin


def normalize_t(t):
    """For a list of arrays, normalize each array to fall between 0 and 1.

    Parameters
    ----------
    t : list of np.ndarray

    Returns
    -------
    list of np.ndarray
    """
    return [(_t - np.min(_t))/(np.max(_t)-np.min(_t)) for _t in t]


def event_hists(data, bins, key='vdw'):
    """Get event histograms. The event sizes and durations are log-distributed; the absolute value of the abscissa
    is taken before the log binning.

    Parameters
    ----------
    data : datautil.SimRun or datautil.SimData
        Data from which to generate histograms
    bins : int
        Number of bins in which the data should be binned

    Returns
    -------
    tuple of np.ndarray
        size_bins, size_hist, time_bins, time_hist
    """
    if isinstance(data, datautil.SimRun) or isinstance(data, datautil.SimData):

        size_bins, size_hist = loghist(data.get_avalanche_sizes(key=key), bins)
        time_bins, time_hist = loghist(data.get_avalanche_durations(), bins)

        return size_bins, size_hist, time_bins, time_hist
    else:
        raise NotImplementedError


def loghist(_data, bins):
    """Generate bins and a histogram, properly normalized by bin size and number of samples, using log spaced bins.
    The absolute value of the data is taken before being binned.

    Parameters
    ----------
    data : np.ndarray
        Data from which to generate histograms
    bins : int
        Number of bins in which the data should be binned

    Returns
    -------
    tuple of np.ndarray
        bins, hist
    """

    data = np.abs(_data)

    logbins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bins)
    hist, _ = np.histogram(data, bins=logbins, density=True)  # density=True apparently makes this a PDF by dividing by the bin width and sample size

    # Normalize the distributions; the number of occurences in each bin is divided by the bin width and the sample size
    # hist = hist/(np.diff(logbins)*len(data))

    # Validate the histograms
    util.validate_pdf(logbins, hist)

    return logbins, hist


def avg_event_size(data, bins=40, key='vdw'):

    sizes = data.get_avalanche_sizes(key=key)
    times = data.get_avalanche_durations()

    log_time_bins = np.logspace(np.log10(np.min(times)), np.log10(np.max(times)), bins+1)  # Bin edges
    avg_size = np.ones(bins)

    for i in range(bins):
        avg_size[i] = np.mean(sizes[np.logical_and(times > log_time_bins[i], times < log_time_bins[i+1])])

    return log_time_bins, avg_size


@nb.jit(nopython=True)
def hist2d(datax, datay, nbinsx, nbinsy):

    binsx = np.linspace(np.min(datax), np.max(datax), nbinsx+1)
    binsy = np.linspace(np.min(datay), np.max(datay), nbinsy+1)

    hist = np.zeros((nbinsx, nbinsy))

    bin_areas = np.outer((binsx[1:] - binsx[:-1]))

    # TODO: Finish this
    raise NotImplementedError


@nb.jit(nopython=True)
def loghist2d(datax, datay, nbinsx, nbinsy):

    datax = np.abs(datax)
    datay = np.abs(datay)

    # These define the bin edges.
    binsx = 10**np.linspace(np.log10(np.min(datax)), np.log10(np.max(datax)), nbinsx+1)
    binsy = 10**np.linspace(np.log10(np.min(datay)), np.log10(np.max(datay)), nbinsy+1)
    hist = np.zeros((nbinsx, nbinsy))

    bin_areas = np.outer((binsx[1:] - binsx[:-1]), (binsy[1:] - binsy[:-1]))

    for i in range(datax.size):

        # Find the correct bin to increment
        _iy = np.nonzero(datay[i] >= binsy[:-1])[0]
        _ix = np.nonzero(datax[i] >= binsx[:-1])[0]

        iy = -1
        ix = -1
        if _iy.size > 0:
            iy = _iy[-1]

        if _ix.size > 0:
            ix = _ix[-1]

        # Increment
        hist[iy, ix] += 1

    hist = hist/(bin_areas*datax.size)

    return hist, binsx, binsy


def joint_pdf_bin_areas(binsx, binsy):
    """Given a 1D set of bin edges along x and y, compute the 2D set of bin areas formed by the grid.

    Parameters
    ----------
    binsx : np.ndarray
        1D array of bin edges
    binsy : np.ndarray
        1D array of bin edges along y

    Returns
    -------
    np.ndarray
        2D array of bin areas for the given bin sizes
    """

    xbsize = binsx[1:] - binsx[:-1]
    ybsize = binsy[1:] - binsy[:-1]
    return np.outer(xbsize, ybsize)


# def joint_pdf_bin_centers(binsx, binsy):
def bin_centers(*bins):
    """Compute the centers of bins given the bin edges.

    Parameters
    ----------
    bins : np.ndarray
        bin edges

    Returns
    -------
    np.ndarray
        bin centers. This array is 1 element shorter than the inputs.
    """
    # return (binsx[1:] + binsx[:-1])*0.5, (binsy[1:] + binsy[:-1])*0.5
    return [(b[1:] + b[:-1])*0.5 for b in bins]


def joint_pdf_mean_y(pdf, binsx, binsy):
    """From a joint PDF, calculate the mean along the y-direction for each x-bin.

    Parameters
    ----------
    pdf : np.ndarray
        Array of shape (binsy.size - 1, binsx.size - 1). This is the probability density function
    binsx : np.ndarray
        Array of bin edges along the x direction
    binsy : np.ndarray
        Array of bin edges along the y direction

    Returns
    -------
    np.ndarray
        The mean along the y-direction for each x-bin; should be of shape (binsx.size - 2).
    """

    # Get the bin centers
    # bincx, bincy = joint_pdf_bin_centers(binsx, binsy)
    bincx, bincy = bin_centers(binsx, binsy)

    # Get the frequency distribution from the probability density by multiplying by bin areas
    freq = pdf*joint_pdf_bin_areas(binsx, binsy)

    # Find the frequency distribution along each column, effectively finding conditional probabilities P(X=x0, Y)
    col_freq = freq/np.outer(np.ones(bincy.size), np.sum(freq, axis=0))

    # Find the average y-value for each column
    col_mean = np.sum(col_freq*np.outer(bincy, np.ones(bincx.size)), axis=0)

    return col_mean


def extent(binsx, binsy):
    """Get the extent of the 2D histogram generated from binsx and binsy.

    Parameters
    ----------
    binsx : np.ndarray
        Bin edges along x
    binsy : np.ndarray
        Bin edges along y

    Returns
    -------
    np.ndarray
        Array of [xmin, xmax, ymin, ymax]
    """
    return np.array([binsx.min(), binsx.max(), binsy.min(), binsy.max()])


def lognan(pdf):
    """Compute the log10 of the input PDF. This function gets around errors associated with taking the log of a
    histogram which has one or more bins with 0 events by masking those bins with np.nan values before taking the log.

    Parameters
    ----------
    pdf : np.ndarray
        Input probability distribution function

    Returns
    -------
    np.ndarray
        Returns log10(pdf), except if there are any bins where the pdf == 0, those bins now have np.nan values.
    """
    _pdf = pdf.copy()
    _pdf[pdf <= 0] = np.nan
    return np.log10(_pdf)


def overhang(wall):
    """Calculate the overhang parameter as a function of y along the wall. See _overhang() for details.

    Parameters
    ----------
    wall : pandas.DataFrame or datautil.DomainWall
        Location of the zero crossing of Mz.

    Returns
    -------
    np.ndarray or list of np.ndarray
        If wall is a single pandas Dataframe (corresponding to zero crossings at a single moment in time), the
        return value is a np.ndarray of values of the overhang as a function of y.

        If wall is a datautil.Domainwall, this is a list of np.ndarrays, each holding the value of the y-dependent
        overhang parameter.
    """

    if isinstance(wall, pd.DataFrame):
        return _overhang(wall)
    elif isinstance(wall, datautil.DomainWall):
        return [_overhang(w) for w in wall]
    else:
        raise ValueError(f'Invalid wall type passed as parameter. type(wall) = {type(wall)}')


def _overhang(wall):
    """Calculate the overhang parameter as a function of y along the wall. For each value of y, the overhang is defined

        δh(y') = max({ZC(x, y=y')}) - min({ZC(x, y=y')})

    where {ZC(x, y)} represents the set of all zero crossings of Mz. If there is only one zero crossing at a given
    y-value, then δh = 0.

    Parameters
    ----------
    wall : pandas.DataFrame
        The locations of the zero crossings of Mz along the wall. Must have 'x' and 'y' columns, but if you've used
        datautil.Domainwall, the columns will be ['x', 'y', 'z', 'mx', 'my', 'mz'].

    Returns
    -------
    np.ndarray
        Overhang as a function of the y-coordinate
    """

    δh = np.empty(len(wall['y'].unique()))
    s_wall = wall.sort_values(['y', 'x'])  # Sort the DataFrame first by 'y', then by 'x'
    for i, val in enumerate(s_wall['y'].unique()):
        row = s_wall.loc[s_wall['y'] == val]['x']
        δh[i] = row.max() - row.min()
    return δh