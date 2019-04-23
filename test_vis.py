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

size = 18
matplotlib.rc('axes', labelsize=size)

# data_lowdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_range/2019-04-17')
# data_highdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.6e-3/2019-04-16')

# times, signals = data_highdmi.events_by_duration(duration=0.5e-9, tol=0.25e-10)
# tbin, sbin = statutil.bin_avg(times, signals, nbins=100)


data_lowdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_range/2019-04-17')

fig, ax = plt.subplots(figsize=(30, 12))
plotutil.burst(ax, data_lowdmi[0], 'viridis')
plt.show()
