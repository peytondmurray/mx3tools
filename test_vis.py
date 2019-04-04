import numpy as np
import matplotlib.pyplot as plt
import mx3tools.datautil as datautil


data = datautil.SimData('/home/pdmurray/Desktop/Workspace/dmidw/coni_ramping/test3/dmidw_ramping_4.mx3',
                        '/home/pdmurray/Desktop/Workspace/dmidw/coni_ramping/test3/dmidw_ramping_4.out')

fig, ax = plt.subplots(figsize=(14, 7))

ax.set_aspect('equal')

ax.set_xlim(0, 1024e-9)
ax.set_ylim(0, 1024e-9)
# ani = data.anim(ax)
ani = data.anim_burst(ax)
plt.show()
