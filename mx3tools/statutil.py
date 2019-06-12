# Methods related to statistical analysis of simulation output files

import numpy as np
import pandas as pd
import numba as nb
import pathlib
import warnings

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
        Signal to be integrated during an avalanche to get avalanche sizes; by default, this is the same signal as
        v, the one being watched for avalanches. But it can also be different, e.g., a domain wall avalanche can
        be thresholded on the velocity, but can be integrated to get the avalanche sizes in terms of Axy and Az.

        If this parameter is applied, no threshold is subtracted from s before it is integrated.

    """

    def __init__(self, t, v, vt, s=None):

        self.t = t
        self.v = v
        self.vt = vt
        self.s = s or v         # If s not passed, use v

        if t.shape != v.shape != s.shape:
            warnings.warn(f't, v, and s must be the same shape. (t, v, s): ({len(t)}, {len(v)}, {len(s)})')
            self.sizes = np.zeros(0)
            self.durations = np.zeros(0)
            return

        self.istart, self.istop = _events(v, vt)
        self.tstart, self.tstop = self.t[self.istart], self.t[self.istop]
        self.durations = self.tstop - self.tstart

        if s is None:
            self.sizes = _event_sizes(self.t, self.v-self.vt, self.istart, self.istop)
        else:
            self.sizes = _event_sizes(self.t, self.s, self.istart, self.istop)

        return


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


def _event_sizes(t, v_minus_vt, i_start, i_stop):
    """Compute the size of each avalanche in the signal. The size of an avalanche is the integral of the signal over
    the time the signal is above the threshold. This integration is done using rectangles, with

        dt[i] = (t[i+1]-t[i-1])/2

    except at the edges, where the forward (for t[0]) or backward (for t[-1]) differences are taken.

    Parameters
    ----------
    t : np.ndarray
        Time
    v : np.ndarray
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
        ret[i] = np.sum(v_minus_vt[i_start[i]:i_stop[i]]*dt[i_start[i]:i_stop[i]], axis=0)

    return ret


def bin_avg(t, s, nbins=None, norm=True):
    """Bin and average the input signals. The times of each event are normalized from 0 to 1.

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
        Number of bins to use. If None, then a sensible number is chosen: t_binned = np.arange(min(t), max(t), mean(dt))
    norm : bool
        Scale the t-axis to [0, 1].

    Returns
    -------
    t_bin : np.ndarray
        Binned timesteps
    s_bin : np.ndarray
        Average value of the signal at each timestep
    """

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
    """Get event histograms. The event sizes and durations are log-distributed.

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


def loghist(data, bins):
    """Generate bins and a histogram, properly normalized by bin size and number of samples, using log spaced bins.

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

    logbins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bins)
    hist, _ = np.histogram(data, bins=logbins)

    # Normalize the distributions; the number of occurences in each bin is divided by the bin width and the sample size
    hist = hist/(np.diff(logbins)*len(data))

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
