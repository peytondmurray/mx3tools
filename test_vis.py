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
import tqdm
import cmocean


data_0 = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.0e-3/2019-05-26/')

bins = 40

fig, ax = plt.subplots(figsize=(10, 10))

sizes = data_0.get_avalanche_sizes(key='vdw')
times = data_0.get_avalanche_durations()

z, xb, yb = statutil.loghist2d(times, sizes, bins, bins)

_z = z.copy()
_z[z <= 0] = np.nan
_z = np.log10(_z)

# mean_Sv = np.mean(z, axis=0)

# im = ax.imshow(_z,
#                origin='lower',
#                extent=[np.min(xb), np.max(xb), np.min(yb), np.max(yb)],
#                vmin=np.min(_z[_z > 0]),
#                cmap=cmocean.cm.ice_r,
#                interpolation='nearest')


# for b in xb:
#     ax.plot([b, b], [np.min(yb), np.max(yb)], '-w', alpha=0.25)
# for b in yb:
#     ax.plot([np.min(xb), np.max(xb)], [b, b],  '-w', alpha=0.25)


_xb_actual = 10**xb
_yb_actual = 10**yb

__xb, __yb = np.meshgrid(_xb_actual, _yb_actual)

_x = (xb[1:]+xb[:-1])*0.5

# _y = np.log10(mean_Sv)

# _y = np.mean((_yb_actual[1:] + _yb_actual[:-1])*0.5*z, axis=0)


probability = (__yb[1:, :-1]-__yb[:-1, :-1])*0.5*(__xb[:-1, 1:]-__xb[:-1, :-1])*0.5*z


im = ax.imshow(probability,
               origin='lower',
               extent=[np.min(xb), np.max(xb), np.min(yb), np.max(yb)],
               vmin=np.min(probability[probability > 0]),
               cmap=cmocean.cm.ice_r,
               interpolation='nearest')



# _y = np.mean((__yb[1:, :-1]-__yb[:-1, :-1])*0.5*(__xb[:-1, 1:]-__xb[:-1, :-1])*0.5*z, axis=0)


# ax.plot((xb[1:]+xb[:-1])*0.5, np.log10(probability), linestyle='-', color='orange')
# ax.plot((xb[1:]+xb[:-1])*0.5, np.log10(_y), linestyle='-', color='orange')
# ax.plot((xb[1:]+xb[:-1])*0.5, np.log10(mean_Sv), linestyle='-', color='orange')
# ax.plot((xb[1:]+xb[:-1])*0.5, np.log10(mean_Sv), linestyle='-', color='orange', drawstyle='steps-mid')

# ax.plot(times, sizes, 'ok', alpha=0.1)
ax.set(xlabel=r'$\log(T_V)$', ylabel=r'$\log(S_V)$', aspect='auto')
# fig.colorbar(im)

plt.show()