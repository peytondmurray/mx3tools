# Methods related to statistical analysis of simulation output files

import numpy as np
import pandas as pd
import numba as nb
import pathlib


class Seismograph:

    def __init__(self, t, v, vt):

        self.t = t
        self.v = v
        self.vt = vt

        self.istart, self.istop = events(v, vt)
        self.tstart, self.tstop = self.t[self.istart], self.t[self.istop]
        self.durations = self.tstart - self.tstop
        self.sizes = event_size(self.t, self.v, self.vt, self.tstart, self.tstop)

        return


def start_indices(v, vt):
    vn = v-vt
    return np.nonzero(np.logical_and(vn[1:] > 0, vn[:-1] < 0))[0]


def end_indices(v, vt):
    vn = v-vt
    return np.nonzero(np.logical_and(vn[1:] < 0, vn[:-1] > 0))[0]


def events(v, vt):
    i_start = start_indices(v, vt)
    i_stop = end_indices(v, vt)

    if i_start[0] > i_stop[0]:
        i_stop = i_stop[1:]
    if i_start[-1] < i_stop[-1]:
        i_stop[-1] = len(v)-1

    return i_start, i_stop


# @nb.jitclass({'start': nb.float32, 'stop': nb.float32, 'size': nb.float32, 'duration': nb.float32})
# class Event:
#     """Holds information about a single event - an excursion of the signal of interest above the threshold value.

#     """

#     def __init__(self, start, stop, size):
#         self.start = start
#         self.stop = stop
#         self.size = size
#         self.duration = self.stop-self.start
#         return


# @nb.jit(nopython=True)
# def next_event(signal, threshold, start):
#     """Return the index of the next rising zero-crossing.

#     Parameters
#     ----------
#     signal : np.ndarray
#         Signal to search for the next zero-crossing
#     threshold : float
#         Signal threshold; index is returned when signal crosses this value
#     start : int
#         Index where the search is started

#     Returns
#     -------
#     int
#         Index of the next event. If no event is found, return the .
#     """

#     for i in range(start, len(signal)):
#         if signal[i] > threshold:
#             return i

#     return len(signal)


# @nb.jit(nopython=True)
# def event_end(signal, threshold, start):

#     for i in range(start, len(signal)):
#         if signal[i] < threshold:
#             return i

#     return len(signal)


@nb.jit(nopython=True)
def event_size(t, signal, threshold, i_start, i_end):

    # V = signal - threshold
    # return np.trapz(y=V[i_start:i_end], x=t[i_start:i_end])

    # Old style. np.trapz gives zero size events if only a single point is above the threshold...
    V = signal[1:-1] - threshold
    dt = (t[2:]-t[:-2])*0.5
    # dt = np.diff(t)
    return np.sum((V*dt)[i_start:i_end])


# @nb.jit(nopython=True)
# def get_events(t, signal, threshold):
#     """Given an input signal time series and threshold, return a list of Events found in the series."""

#     events = []

#     i = 0
#     while i < len(signal):
#         i_start = next_event(signal, threshold, i)

#         if i_start != len(signal):
#             i_end = event_end(signal, threshold, i_start)
#             size = event_size(t, signal, threshold, i_start, i_end)
#             events.append(Event(t[i_start], t[i_end], size))
#             i = i_end
#         else:
#             i = i_start

#     return events
