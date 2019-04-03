# Methods related to statistical analysis of simulation output files

import numpy as np
import pandas as pd
import numba as nb
import pathlib


@nb.jitclass({'start': nb.float32, 'stop': nb.float32, 'size': nb.float32, 'duration': nb.float32})
class Event:
    """Holds information about a single event - an excursion of the signal of interest above the threshold value.

    """

    def __init__(self, start, stop, size):
        self.start = start
        self.stop = stop
        self.size = size
        self.duration = self.stop-self.start
        return


@nb.jit(nopython=True)
def next_event(signal, threshold, start):
    """Return the index of the next rising zero-crossing.

    Parameters
    ----------
    signal : np.ndarray
        Signal to search for the next zero-crossing
    threshold : float
        Signal threshold; index is returned when signal crosses this value
    start : int
        Index where the search is started

    Returns
    -------
    int
        Index of the next event. If no event is found, return the .
    """

    for i in range(start, len(signal)):
        if signal[i] > threshold:
            return i

    return len(signal)


@nb.jit(nopython=True)
def event_end(signal, threshold, start):

    for i in range(start, len(signal)):
        if signal[i] < threshold:
            return i

    return len(signal)


@nb.jit(nopython=True)
def event_size(t, signal, threshold, i_start, i_end):

    V = signal - threshold
    return np.trapz(y=V[i_start:i_end], x=t[i_start:i_end])

    # Old style. Numba now supports nb.trapz, so there isn't much point here.
    # dt = t[1:]-t[:-1]
    # dt = np.diff(t)
    # return np.sum((V[:-1]*dt)[i_start:i_end])


@nb.jit(nopython=True)
def get_events(t, signal, threshold):
    """Given an input signal time series and threshold, return a list of Events found in the series."""

    events = []

    i = 0
    while i < len(signal):
        i_start = next_event(signal, threshold, i)

        if i_start != len(signal):
            i_end = event_end(signal, threshold, i_start)
            size = event_size(t, signal, threshold, i_start, i_end)
            events.append(Event(t[i_start], t[i_end], size))
            i = i_end
        else:
            i = i_start

    return events


def get_sizes(events):
    return np.array([event.size for event in events])


def get_durations(events):
    return np.array([event.duration for event in events])
