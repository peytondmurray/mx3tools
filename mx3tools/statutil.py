# Methods related to statistical analysis of simulation output files

import numpy as np
import pandas as pd
import numba as nb
import pathlib


class Seismograph:
    """Finds the avalanches and corresponding sizes and durations in a signal.
    """

    def __init__(self, t, v, vt):

        self.t = t
        self.v = v
        self.vt = vt

        self.istart, self.istop = _events(v, vt)
        self.tstart, self.tstop = self.t[self.istart], self.t[self.istop]
        self.durations = self.tstop - self.tstart
        self.sizes = _event_sizes(self.t, self.v, self.vt, self.istart, self.istop)

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

    return np.nonzero(np.logical_and(v[1:] > vt, v[:-1] < vt))[0]+1


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

    return np.nonzero(np.logical_and(v[1:] < vt, v[:-1] > vt))[0]+1


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
        if i_start[-1] != len(v) - 1:
            i_stop = np.append(i_stop, len(v)-1)
        else:
            i_start = i_start[:-1]

    return i_start, i_stop


def _event_sizes(t, v, vt, i_start, i_stop):
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
    vt : float
        Threshold signal value
    i_start : np.ndarray
        Array of starting indices of the events
    i_stop : np.ndarray
        Array of ending indices of the events

    Returns
    -------
    np.ndarray
        Size of each event
    """

    V = v - vt
    dt = np.hstack((np.array([t[1] - t[0]]), (t[2:]-t[:-2])*0.5, np.array([t[-1]-t[-2]])))

    ret = np.empty(i_start.shape[0])
    for i in range(len(i_start)):
        ret[i] = np.sum(V[i_start[i]:i_stop[i]]*dt[i_start[i]:i_stop[i]], axis=0)

    return ret
