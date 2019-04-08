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
        self.durations = self.tstop - self.tstart
        self.sizes = event_sizes(self.t, self.v, self.vt, self.istart, self.istop)

        return


def start_indices(v, vt):
    return np.nonzero(np.logical_and(v[1:] > vt, v[:-1] < vt))[0]+1


def end_indices(v, vt):
    return np.nonzero(np.logical_and(v[1:] < vt, v[:-1] > vt))[0]+1


def events(v, vt):
    i_start = start_indices(v, vt)
    i_stop = end_indices(v, vt)

    if i_start[0] > i_stop[0]:
        i_stop = i_stop[1:]
    if i_start[-1] > i_stop[-1]:
        if i_start[-1] != len(v) - 1:
            i_stop = np.append(i_stop, len(v)-1)
        else:
            i_start = i_start[:-1]

    return i_start, i_stop


def event_sizes(t, signal, threshold, i_start, i_stop):

    # Old style. np.trapz gives zero size events if only a single point is above the threshold...
    V = signal - threshold
    dt = np.hstack((np.array([t[1] - t[0]]), (t[2:]-t[:-2])*0.5, np.array([t[-1]-t[-2]])))

    ret = np.empty(i_start.shape[0])
    for i in range(len(i_start)):
        ret[i] = np.sum(V[i_start[i]:i_stop[i]]*dt[i_start[i]:i_stop[i]], axis=0)

    return ret
