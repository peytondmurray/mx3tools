
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.animation as animation
import mx3tools.ovftools as ovftools
import mx3tools.statutil as statutil
import mx3tools.datautil as datautil
import mx3tools.plotutil as plotutil
import mx3tools.util as util
import tqdm
import cmocean
import pprint
plt.style.use('dark_background')
np.set_printoptions(linewidth=120, precision=3)


data = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.0e-3/2019-06-24/')

# azbins, azhist, _, _ = statutil.event_hists(data, 40, key='Az')
# print('Done!')


d = data.get_sim(1)

y = util.diff(d.t(), d.table['ext_exactdwposavg (m)'].values, 1, 1)

plt.plot(d.t(), d.vdw(), '-w')
plt.plot(d.t(), y, '-r')
plt.tight_layout()
plt.show()
