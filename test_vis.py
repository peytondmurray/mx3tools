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
# data_highdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.6e-3/2019-04-16')
data_highdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/parameter_space/coni_ramping/test8_mzAvg')

fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(8, 16))
plotutil.burst(ax, data_highdmi[0], cmap='angle')

plt.tight_layout()

# plt.show()
plt.savefig('/home/pdmurray/Desktop/test8_mzAvg.png', dpi=300)
