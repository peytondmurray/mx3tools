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
        self.sizes = event_sizes(self.t, self.v, self.vt, self.tstart, self.tstop)

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
    if i_start[-1] > i_stop[-1]:
        i_stop = np.append(i_stop, len(v)-1)

    return i_start, i_stop


# @nb.jit(nopython=True)
def event_sizes(t, signal, threshold, i_start, i_end):

    # V = signal - threshold
    # return np.trapz(y=V[i_start:i_end], x=t[i_start:i_end])

    # Old style. np.trapz gives zero size events if only a single point is above the threshold...
    V = signal - threshold
    dt = np.hstack((np.array([t[1] - t[0]]), (t[2:]-t[:-2])*0.5, np.array([t[-1]-t[-2]])))

    # TODO: i_start and i_end are arrays. Need to find all event sizes at once.

    return np.sum((V*dt)[i_start:i_end])
