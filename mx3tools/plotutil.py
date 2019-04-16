# Plotting utilities for output of simulation files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.collections as mplcollections
import matplotlib.animation as animation
import matplotlib.patches as mplp
import matplotlib.colors as mplcolors
from . import datautil


def plot_dw(data, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()

    data = data.sort_values('y')

    ax.plot(data['x'], data['y'], **kwargs)
    ax.set_xlim(0, 2*data['x'].max())
    ax.set_ylim(0, data['y'].max())
    ax.set_aspect('equal')
    plt.draw()
    return


def plot_dw_config(data, ax=None, cmap='twilight', marker='cell', dx=2e-9):
    """Plot a domain wall. In-plane magnetization (xy) is encoded in the color of the plot.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe: [x, y, z, mx, my, mz]
    ax : Axes, optional
        Axes on which to draw the wall. If None, a new figure is generated.
    cmap : str
        Colormap to use for plotting the magnetization along the DW. Use a cyclic colormap; twilight is default.
    marker : str
        'cell' or 'line'. 'cell' draws rectangles to represent the domain wall; this is in some sense the best option,
        since the simulation consists of cells.

        'line' connects the locations of the domain wall using a line.
    dx : float
        If 'cell' markers are used, this specifies the x and y sizes of each cell.

    """

    if ax is None:
        _, ax = plt.subplots()
    cmap = mplcm.get_cmap(cmap)
    colors = []

    data = data.sort_values('y')

    if marker == 'line':
        segments = []

        for i in range(len(data)-1):
            segments.append(((data.iloc[i]['x'], data.iloc[i]['y']), (data.iloc[i+1]['x'], data.iloc[i+1]['y'])))
            colors.append(cmap(np.arctan2(data.iloc[i]['my'], data.iloc[i]['mx'])/(2*np.pi) + 0.5))

        collection = mplcollections.LineCollection(segments=segments, colors=colors, linewidths=6)
        collection.set_capstyle('round')

        ax.add_collection(collection)

    elif marker == 'cell':
        for i in range(len(data)-1):
            color = cmap(np.arctan2(data.iloc[i]['my'], data.iloc[i]['mx'])/(2*np.pi) + 0.5)
            ax.add_patch(mplp.Rectangle((data.iloc[i]['x'], data.iloc[i]['y']),
                                        dx,
                                        dx,
                                        edgecolor='none',
                                        facecolor=color))

    ax.set_xlim(0, 0.25*data['y'].max())
    ax.set_ylim(0, data['y'].max())
    ax.set_aspect('equal')
    plt.draw()
    return


def color_wheel(fig, cmap='twilight'):

    # Generate a figure with a polar projection
    ax = fig.add_axes([0.2, 0.2, 0.1, 0.1], projection='polar')

    # Define colormap normalization for 0 to 2*pi
    norm = mplcolors.Normalize(0, 2*np.pi)

    # Plot a color mesh on the polar plot with the color set by the angle
    n = 200                                                 # the number of secants for the mesh
    t = np.linspace(0, 2*np.pi, n)                          # theta values
    r = np.linspace(.6, 1, 2)                               # raidus values change 0.6 to 0 for full circle
    _, tg = np.meshgrid(r, t)                               # create a r,theta meshgrid
    c = tg                                                  # define color values as theta value
    ax.pcolormesh(t, r, c.T, norm=norm, cmap='twilight')    # plot the colormesh on axis with colormap
    ax.set_yticklabels([])                                  # turn of radial tick labels (yticks)
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)                   # turn off the axis spine.

    return


def plot_hists(axes, data=None, bins=100, **kwargs):

    if isinstance(data, datautil.SimRun) or isinstance(data, datautil.SimData):
        sizes = data.get_avalanche_sizes()
        durations = data.get_avalanche_durations()
    else:
        raise NotImplementedError

    log_size_bins = np.logspace(np.log10(np.min(sizes)), np.log10(np.max(sizes)), bins)
    log_duration_bins = np.logspace(np.log10(np.min(durations)), np.log10(np.max(durations)), bins)

    sizes_hist, _ = np.histogram(sizes, bins=log_size_bins)
    durations_hist, _ = np.histogram(durations, bins=log_duration_bins)

    # axes[0].plot(log_size_bins[:-1], sizes_hist, '-or')
    # axes[1].plot(log_duration_bins[:-1], durations_hist, '-or')

    # axes[0].step(log_size_bins[:-1], sizes_hist, color=kwargs.get('color', 'r'), where='post')
    # axes[1].step(log_duration_bins[:-1], durations_hist, color=kwargs.get('color', 'r'), where='post')

    axes[0].fill_between(log_size_bins[:-1],
                         sizes_hist,
                         facecolor=kwargs.get('facecolor', 'dodgerblue'),
                         step='post',
                         edgecolor=kwargs.get('edgecolor', 'dodgerblue'))
    axes[1].fill_between(log_duration_bins[:-1],
                         durations_hist,
                         facecolor=kwargs.get('facecolor', 'dodgerblue'),
                         step='post',
                         edgecolor=kwargs.get('edgecolor', 'dodgerblue'))

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    axes[0].set_xlabel('Size')
    axes[1].set_xlabel('Duration')
    axes[0].set_ylabel('Frequency')
    axes[1].set_ylabel('Frequency')

    return


def anim_track(ax, *walls, interval=100, maxframes=None, label='time', **kwargs):

    lines = []
    for wall in walls:
        lines.append(ax.plot(wall[0]['x'], wall[0]['y'], **kwargs)[0])
    tag = ax.text(kwargs.get('textx', 0.1), kwargs.get('texty', 0.1), '', transform=ax.transAxes)

    # ax.set_xlim(kwargs.get('xlim', (0, np.max(wall.get_window_pos()))))

    ax.set_aspect('equal')

    def init():
        for line in lines:
            line.set_xdata([])
            line.set_ydata([])

        tag.set_text('')

        return

    def anim(i):
        for wall, line in zip(walls, lines):
            line.set_xdata(wall[i]['x'])
            line.set_ydata(wall[i]['y'])

        if label == 'time':
            tag.set_text(f'Time: {wall.time[i]:3.3e}')
        elif label == 'iteration':
            tag.set_text(f'Frame: {i}')
        minX = np.min([wall[i]['x'].min() for wall in walls])
        maxX = np.max([wall[i]['x'].max() for wall in walls])
        span = maxX-minX
        ax.set_xlim(minX-0.1*span, maxX+0.1*span)
        return

    return animation.FuncAnimation(ax.get_figure(),
                                   func=anim,
                                   init_func=init,
                                   frames=len(wall)-2 if maxframes is None else maxframes,
                                   interval=interval,
                                   blit=False)