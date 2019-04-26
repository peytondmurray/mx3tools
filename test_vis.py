import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.animation as animation
import mx3tools.ovftools as ovftools
import mx3tools.statutil as statutil
import mx3tools.datautil as datautil
import mx3tools.plotutil as plotutil


def mz_to_pos_in_window(mz):
    return (1 - (1 - mz)/2)*1024*2e-9


size = 18
matplotlib.rc('axes', labelsize=size)

# data_lowdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_range/2019-04-17')
data_highdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.6e-3/2019-04-16')
# data_highdmi = datautil.SimData(script='', data_dir='/home/pdmurray/Desktop/Workspace/dmidw/barkhausen_0.out')

# times, signals = data_highdmi.events_by_duration(duration=0.5e-9, tol=0.25e-10)
# tbin, sbin = statutil.bin_avg(times, signals, nbins=None)


# data_lowdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_range/2019-04-17')

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(15, 8))
# plotutil.burst(ax, data_lowdmi[0], 'viridis')

# for _t, _s in zip(times, signals):
#     plt.plot(_t, _s, '-', color='dodgerblue', alpha=0.2)

# plt.plot(tbin, sbin, '-k')

# plotutil.sanity_event_shape(ax, data_highdmi[0], 0.5e-9, 0.25e-10)
# plotutil.event_shape(ax, data_highdmi, 0.5e-9, 0.25e-10)


# plotutil.plot_dt(ax, data_highdmi[0])

ax[0].plot(data_highdmi[0].t(), data_highdmi[0].vdw(), '-k')
ax[1].plot(data_highdmi[0].t(), data_highdmi[0].dwpos(), '-r')
ax[2].plot(data_highdmi[0].t(), data_highdmi[0].shift(), '-b')
ax[3].plot(data_highdmi[0].t(), mz_to_pos_in_window(data_highdmi[0].table['mz ()']), '-g')

plt.tight_layout()

plt.show()
