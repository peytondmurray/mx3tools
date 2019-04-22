import numpy as np
import matplotlib.pyplot as plt
import mx3tools.datautil as datautil
import mx3tools.plotutil as plotutil


# data = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/coni_ramping/test5/')

# fig, ax = plt.subplots(figsize=(10, 10))

# ani = data[0].anim(ax)

# data = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/coni_ramping/test8/')
# wall = data[0].get_wall()
# sizes = data.get_avalanche_sizes()

data_lowdmi = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.6e-3/2019-04-16')


fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
plotutil.plot_hists(ax, data=data_lowdmi, bins=40, duration_units='min', facecolor='#474747', edgecolor='#474747')
