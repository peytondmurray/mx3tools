import numpy as np
import matplotlib.pyplot as plt
import mx3tools.datautil as datautil


# data = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/coni_ramping/test5/')

# fig, ax = plt.subplots(figsize=(10, 10))

# ani = data[0].anim(ax)

data = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/coni_ramping/test8/')
wall = data[0].get_wall()
# sizes = data.get_avalanche_sizes()
