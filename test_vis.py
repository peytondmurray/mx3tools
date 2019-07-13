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

# data_0 = datautil.SimRun('/mnt/Data/dmidw/D_0.0e-3/2019-05-26/')
df = pd.read_csv('/mnt/Data/dmidw/D_0.0e-3/2019-05-26/barkhausen_0.out/domainwall006000.csv', skiprows=3)

# datautil.n_bloch_lines(df, 2e-9)

# f = datautil.wrap_distance(df.iloc[0], df.iloc[1:], df['y'].max())

# blochlines = datautil.n_bloch_lines(df, 2e-9)
_df = datautil.tsp(df, 0, df['y'].max(), tmax=5, nt=1000)

# print('a')

fig, ax = plt.subplots(figsize=(10, 10))


# x = df['x']
# y = df['y']
x = _df['x']
y = _df['y']

# ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
ax.plot(x, y, '-w', alpha=0.5)
ax.scatter(x, y, c=range(len(x)), cmap='viridis', marker='.')
ax.set(aspect='equal', ylim=(0, df['y'].max()), xlim=(df['x'].min(), df['x'].max()))
plt.show()
