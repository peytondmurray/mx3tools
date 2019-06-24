# Plotting utilities for output of simulation files

import ipywidgets as ipw
import IPython.display as ipd
import re
import warnings
import tqdm
import numpy as np
import pandas as pd
import scipy.interpolate as sci
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mplcollections
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.colors as mplcolors
import PIL.Image as Image
import cv2
from . import datautil
from . import util
from . import statutil
from . import ovftools
from . import ioutil
import cmocean as cmo


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
    cmap = cm.get_cmap(cmap)
    colors = []

    # TODO figure out how to deal with walls which are out of order
    # data = data.sort_values('y')

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
            ax.add_patch(patches.Rectangle((data.iloc[i]['x'], data.iloc[i]['y']),
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


def plot_hists(axes, data, bins=40, tunits='ns', sunits='nm', key='vdw', **kwargs):
    """Plot the t and s histograms. See plot_s_hist and plot_t_hist docstrings for more info.

    Parameters
    ----------
    axes : axes
        Tuple of axes on which to plot the t and s histograms, respectively
    data : mx3tools.SimData or mx3tools.SimRun
        Data from which the histograms are generated
    """

    sbins, shist, tbins, thist = statutil.event_hists(data, bins, key=key)
    plot_t_hist(axes[0], tbins, thist, tunits=tunits, **kwargs)
    plot_s_hist(axes[1], sbins, shist, sunits=sunits, **kwargs)
    return


def plot_s_hist(ax, sbins, shist, sunits='nm', **kwargs):
    """Plot the s histogram.

    Parameters
    ----------
    ax : Axes
        Axes on which to plot the histogram
    sbins : np.ndarray
        The edges of the s histogram bins, including the rightmost edge; len(sbins) = len(shist) + 1
    shist : np.ndarray
        The s histogram; shist[i] gives the value of the histogram between sbins[i] and sbins[i+1]
    sunits : str, optional
        Units to be used for plotting the s values: 'nm' [default] or 'm', or pass your own float
    """

    if sbins is not None and shist is not None:
        if isinstance(sunits, str):
            if sunits == 'nm':
                sbins /= 1e-9
            elif sunits == 'm':
                pass
            else:
                raise NotImplementedError
        elif isinstance(sunits, float):
            sbins /= sunits
        else:
            raise NotImplementedError

        # Make a shallow copy, so that when we pop from the kwargs we don't modify them (dicts are mutable, so we
        # wouldn't be able to use the same kwargs in any other function afterwards if we don't copy here)
        kwargs = kwargs.copy()
        fc = kwargs.pop('facecolor', 'dodgerblue')
        ec = kwargs.pop('edgecolor', 'dodgerblue')

        # Matplotlib doesn't close the last bin properly sometimes (?). Append on a zero to the y-values as workaround.
        ax.fill_between(sbins, np.hstack((shist, np.zeros(1))), facecolor=fc, step='post', edgecolor=ec)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(f'Size ({sunits})')
        ax.set_ylabel('Frequency')

        return
    else:
        raise ValueError('tbins and thist both must be passed as parameters.')


def plot_t_hist(ax, tbins=None, thist=None, tunits='ns', **kwargs):
    """Plot the t histogram.

    Parameters
    ----------
    ax : Axes
        Axes on which to plot the histogram
    tbins : np.ndarray
        The edges of the t histogram bins, including the rightmost edge; len(tbins) = len(thist) + 1
    thist : np.ndarray
        The t histogram; thist[i] gives the value of the histogram between tbins[i] and tbins[i+1]
    tunits : str or float
        Units to be used for plotting the t values: 'ns' [default] or 's', or pass your own float
    """

    if tbins is not None and thist is not None:
        if isinstance(tunits, str):
            if tunits == 'ns':
                tbins /= 1e-9
            elif tunits == 's':
                pass
            else:
                raise NotImplementedError
        elif isinstance(tunits, float):
            tbins /= tunits
        else:
            raise NotImplementedError

        # Make a shallow copy, so that when we pop from the kwargs we don't modify them (dicts are mutable, so we
        # wouldn't be able to use the same kwargs in any other function afterwards if we don't copy here)
        kwargs = kwargs.copy()
        fc = kwargs.pop('facecolor', 'dodgerblue')
        ec = kwargs.pop('edgecolor', 'dodgerblue')

        # Matplotlib doesn't close the last bin properly sometimes (?). Append on a zero to the y-values as workaround.
        ax.fill_between(tbins, np.hstack((thist, np.zeros(1))), facecolor=fc, step='post', edgecolor=ec, **kwargs)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(f'Duration ({tunits})')
        ax.set_ylabel('Frequency')
        return
    else:
        raise ValueError('tbins and thist both must be passed as parameters.')


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


def vdw(ax, data, Dind='', size=18, show_D=True, **kwargs):
    ax.plot(data.t()/1e-9, data.vdw(), **kwargs, label='$V_{DW}$')
    if show_D:
        ax.text(0.2, 0.5, f'$D = {Dind}$'+' J/m$^2$', transform=ax.transAxes, size=size)
    ax.set_ylabel(r'$V_{DW}$ (m/s)')
    ax.set_xlabel(r'$t$ (ns)')
    return


def axyz(ax, data, ls_axy='-', ls_az='-', c_axy='r', c_az='dodgerblue'):

    ax.plot(data.t()/1e-9, data.Axy(), linestyle=ls_axy, color=c_axy, label=r'$A_{xy}$')
    ax.plot(data.t()/1e-9, data.Az(), linestyle=ls_az, color=c_az, label=r'$A_z$')
    ax.set_ylabel(r'$A$ (rad/s)')
    ax.set_xlabel(r'$t$ (ns)')
    return


def burst(ax, data, cmap='viridis', **kwargs):

    ax.set_aspect('equal')

    wall = data.get_wall()

    if cmap is None:
        segments = []
        for i in range(len(wall)):
            # ax.plot(wall.config[i]['x'], wall.config[i]['y'], kwargs.get('color', 'k'), linestyle='-')
            segments += list(zip(wall.config[i]['x'], wall.config[i]['y']))

        collection = mplcollections.LineCollection(segments=segments, colors=kwargs.get('color', 'k'), linestyles='-')
        ax.add_collection(collection)

    elif cmap == 'angle':
        warnings.warn('Burst with cmap=angle is slow. Matplotlib cannot handle this many lines; ctrl-c to give up.')
        for w in tqdm.tqdm(wall.config, desc='Plotting DW configs'):
            plot_dw_config(w, ax=ax, cmap='twilight', marker='line')

    elif isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

        polygons = []
        segments = []
        for i in range(1, len(wall)):
            old_wall = np.array(sorted(zip(wall.config[i-1]['x'].values, wall.config[i-1]['y'].values),
                                       key=lambda a: a[1]))
            new_wall = np.flipud(np.array(sorted(zip(wall.config[i]['x'].values, wall.config[i]['y'].values),
                                                 key=lambda a: a[1])))
            polygon_edge = np.vstack((old_wall, new_wall))
            polygons.append(patches.Polygon(polygon_edge))

            # Highlight the edge with a faint black outline
            plt.plot(polygon_edge[:, 0], polygon_edge[:, 1], '-k', alpha=0.3, linewidth=1)
            segments.append(polygon_edge)

        polygon_collection = mplcollections.PatchCollection(polygons[::-1], edgecolors='',
                                                            facecolors=cmap(np.linspace(0, 1, len(polygons))))
        line_collection = mplcollections.LineCollection(segments=segments, colors='k', linewidths=1, alpha=0.3)

        ax.add_collection(polygon_collection)
        ax.add_collection(line_collection)

    else:
        raise ValueError(f'Invalid cmap argument: {cmap}')

    return


def anim(ax, data, track=False, **kwargs):

    if not isinstance(data, datautil.SimData):
        raise NotImplementedError

    ax.set_aspect('equal')
    wall = data.get_wall()

    line = ax.plot(wall.config[0]['x'], wall.config[0]['y'], color=kwargs.get('color', 'k'), linestyle='-')[0]
    tag = ax.text(kwargs.get('textx', 0.1), kwargs.get('texty', 0.1), '', transform=ax.transAxes)

    def init():
        ax.plot(wall.config[0]['x'], wall.config[0]['y'], color=kwargs.get('color', 'k'), linestyle='-')
        ax.plot(wall.config[0]['x'], wall.config[0]['y'], color=kwargs.get('color', 'k'), linestyle='-')
        tag.set_text('')
        return

    if track:

        def anim(i):
            xlim = kwargs.get('xlim', (0, 1024e-9))
            line.set_xdata(wall.config[i]['x'])
            line.set_ydata(wall.config[i]['y'])
            tag.set_text(f'Time: {wall.time[i]:3.3e}')
            ax.set_xlim(np.array(xlim)-np.mean(xlim)+np.mean(wall.config[i]['x']))
            return

    else:
        ax.set_xlim(kwargs.get('xlim', (0, np.max(wall.get_window_pos()))))

        def anim(i):
            line.set_xdata(wall.config[i]['x'])
            line.set_ydata(wall.config[i]['y'])
            tag.set_text(f'Time: {wall.time[i]:3.3e}')
            return

    return animation.FuncAnimation(ax.get_figure(),
                                   func=anim,
                                   init_func=init,
                                   frames=kwargs.get('maxframes', len(wall)-2),
                                   interval=kwargs.get('interval', 100),
                                   blit=False)


def anim_burst(ax, data, cmap, track=False, **kwargs):

    if not isinstance(data, datautil.SimData):
        raise NotImplementedError

    ax.set_aspect('equal')
    wall = data.get_wall()

    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    tag = ax.text(kwargs.get('textx', 0.1), kwargs.get('texty', 0.1), '', transform=ax.transAxes)

    def init():
        ax.plot(wall.config[0]['x'], wall.config[0]['y'], color=kwargs.get('color', 'k'), linestyle='-')
        ax.plot(wall.config[0]['x'], wall.config[0]['y'], color=kwargs.get('color', 'k'), linestyle='-')
        tag.set_text('')
        return

    if track:
        xlim = kwargs.get('xlim', (0, 1024e-9))

        def anim(i):
            ax.plot(wall.config[i+1]['x'], wall.config[i+1]['y'], color=kwargs.get('color', 'k'), linestyle='-')
            ax.fill_betweenx(wall.config[i]['y'],
                             wall.config[i]['x'],
                             wall.config[i+1]['x'],
                             facecolor=cmap(wall.time[i]/wall.time[-1]))
            tag.set_text(f'Time: {wall.time[i+1]:3.3e}')
            ax.set_xlim(np.array(xlim)-np.mean(xlim)+np.mean(wall.config[i]['x']))
            return
    else:
        ax.set_xlim(kwargs.get('xlim', (0, np.max(wall.get_window_pos()))))

        def anim(i):
            ax.plot(wall.config[i+1]['x'], wall.config[i+1]['y'], color=kwargs.get('color', 'k'), linestyle='-')
            ax.fill_betweenx(wall.config[i]['y'],
                             wall.config[i]['x'],
                             wall.config[i+1]['x'],
                             facecolor=cmap(wall.time[i]/wall.time[-1]))
            tag.set_text(f'Time: {wall.time[i+1]:3.3e}')
            return

    return animation.FuncAnimation(ax.get_figure(),
                                   func=anim,
                                   init_func=init,
                                   frames=len(wall)-2,
                                   interval=kwargs.get('interval', 100))


def event_shape(ax, data, duration, tol, nev_text=True, **kwargs):
    """Plot the average event shape, including contributions from events with duration D such that

            duration-tol < D < duration+tol

    Parameters
    ----------
    ax : Axes
        Axes on which to draw the event shape
    data : datautil.SimRun or datautil.SimData
        Data from which to get the events
    duration : float
        Duration of events to include in average
    tol : float
        Tolerance for determining which events to include
    nev_text : bool, optional
        If True, the number of events in drawn on the axes, by default True

    """

    t, s, tbin, sbin = statutil.bin_avg_event_shape(data, duration, tol)
    ax.plot(tbin, sbin, '-', color='dodgerblue', linewidth=3)
    if nev_text:
        ax.text(0.1, 0.8, f'# events: {len(t)}', transform=ax.transAxes, color='w')

    return


def plot_individual_events(ax, data, duration, tol, nev_text=True, **kwargs):
    """Plot all the individual events in the dataset, including contributions from events with duration D such that

            duration-tol < D < duration+tol

    Parameters
    ----------
    ax : Axes
        Axes on which to draw the events
    data : datautil.SimRun or datautil.SimData
        Data from which to get the events
    duration : float
        Duration of events to draw
    tol : float
        Tolerance for determining which events to include
    """

    t, s = data.events_by_duration(duration, tol)
    t = statutil.normalize_t(t)

    lines = []
    for _t, _s in zip(t, s):
        lines.append(list(zip(_t, _s)))

    # Default values
    kwargs = util.dict_add(kwargs, {'colors': 'k', 'alpha': 0.2})

    ax.add_collection(mplcollections.LineCollection(lines, **kwargs))

    if nev_text:
        ax.text(0.1, 0.8, f'# events: {len(t)}', transform=ax.transAxes, color='w')

    return


def sanity_event_shape(ax, data, duration, tol, **kwargs):
    t, s = data.events_by_duration(duration, tol)

    ax.plot(data.t(), data.vdw(), '-k')

    # lines = []
    for _t, _s in zip(t, s):
        # lines.append(list(zip(_t, _s)))
        ax.plot(_t, _s, '-r')

    # Default values
    kwargs = util.dict_add(kwargs, {'colors': 'r', 'linestyles': 'solid'})

    # ax.add_collection(mplcollections.LineCollection(lines, **kwargs))

    ax.set_xlim(np.min(data.t()), np.max(data.t()))

    return


def plot_dt(ax, data, **kwargs):
    kwargs = util.dict_add(kwargs, {'color': 'r', 'linestyle': '-'})
    ax.plot(np.diff(data.t()), **kwargs)
    return


def spacetime_wall(ax, data, **kwargs):

    wall = data.get_wall()
    maxlen = np.max([len(w) for w in wall.config])
    wall_grid = np.empty((len(wall), maxlen))
    for i in range(len(wall)):
        _interp_x = sci.interp1d(np.linspace(0, 1, len(wall[i])), wall[i]['mx'])
        _interp_y = sci.interp1d(np.linspace(0, 1, len(wall[i])), wall[i]['my'])
        wall_grid[i] = np.arctan2(_interp_y(np.linspace(0, 1, maxlen)), _interp_x(np.linspace(0, 1, maxlen)))

    return wall_grid


def ovfwidget(first_file, cfunc=2, cmap='viridis', norm=True, logabs=False, scale=0.5):
    """When called in a notebook, make an interactive ipywidget slider. When you move the slider, you can view
    the different input ovf vector files as images generated by PIL.

    Parameters
    ----------
    first_file : str
        First file in the set of ovf files to show
    cfunc : int or function
        Component of the vector field to be shown, or function which takes an array of data[:, 0, :, :, :] and returns
        a 3D array: ret[:, :, :]
    cmap : str or LinearSegmentedColormap
        Colormap to use
    norm : bool
        When True, normalize the values in the data across all files/time steps
    logabs : bool
        Show the log10(abs()) of the data
    scale : float
        Scale the output images, adjust as necessary to fit your notebook
    """
    first_file = ioutil.pathize(first_file)

    if isinstance(cfunc, int):
        data = ovftools.group_unpack(first_file)[:, 0, :, :, cfunc]
    else:
        data = cfunc(ovftools.group_unpack(first_file)[:, 0, :, :, :])

    if isinstance(cmap, str):
        if cmap in plt.colormaps():
            cmap = cm.get_cmap(cmap)
        if cmap in cmo.cm.cmapnames:
            cmap = getattr(cmo.cm, cmap)
        else:
            raise ValueError(f'cmap {cmap} not recognized by either matplotlib or cmocean.')

    if logabs:
        data = np.log10(np.abs(data))

    if norm:
        data = (data - data[data > -np.inf].min())/(data[data > -np.inf].max() - data[data > -np.inf].min())

    def _show(frame=(0, len(data)-1)):
        im = Image.fromarray(np.uint8(cmap(data[frame])*255))
        im.thumbnail((scale*np.array(im.size)).astype(int))
        ipd.display(im)
        return im

    interactive_plot = ipw.interactive(_show)
    output = interactive_plot.children[-1]
    output.layout.height = f'{scale*data.shape[1]}px'

    return interactive_plot


def ovfVideo(first_file, fname='out.avi', fps=30, comp=2, cmap='viridis', norm=True, logabs=False, scale=0.5):
    first_file = ioutil.pathize(first_file)
    data = ovftools.group_unpack(first_file)[:, 0, :, :, comp]

    if logabs:
        data = np.log10(np.abs(data))

    if norm:
        data = (data - data[data > -np.inf].min())/(data[data > -np.inf].max() - data[data > -np.inf].min())

    writer = cv2.VideoWriter(fname,
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             fps,
                             (data.shape[2], data.shape[1]),
                             True)

    data = (np.uint8(255*cm.get_cmap(cmap)(data))[:, :, :, :-1])[:, :, :, ::-1]

    for frame in tqdm.tqdm(range(data.shape[0]), desc='Writing video'):
        writer.write(data[frame])

    cv2.destroyAllWindows()
    writer.release()

    return


def bin_edge_lines(ax, xb, yb, **kwargs):

    lines = []
    for _xb in xb:
        lines.append([(_xb, yb.min()), (_xb, yb.max())])
    for _yb in yb:
        lines.append([(xb.min(), _yb), (xb.max(), _yb)])

    collection = mplcollections.LineCollection(segments=lines, **kwargs)
    ax.add_collection(collection)

    return
