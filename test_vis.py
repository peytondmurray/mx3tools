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
plt.style.use('dark_background')


data = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.0e-3/2019-06-24/')

azbins, azhist, _, _ = statutil.event_hists(data, 40, key='Az')
print('Done!')