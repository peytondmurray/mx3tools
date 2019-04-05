# Data structures related to simulation output files

import re
import pathlib
import warnings
import numpy as np
import pandas as pd
import astropy.stats as aps
import scipy.signal as scs
import scipy.constants as scc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from . import statutil
from . import ioutil


class DomainWall:

    def __init__(self, root):
        self.config = []
        self.time = []
        self.root = root
        self.window_pos = []

        files = []
        for item in self.root.iterdir():
            if re.search(r'domainwall\d{6}.csv', item.name) is not None:
                files.append(self.root / item.name)

        if len(files) == 0:
            raise ValueError('No domain wall files found.')

        files = sorted(files)
        for item in files:
            self.append(item)

        return

    def append(self, fname):
        with open(fname, 'r') as f:
            self.time.append(float(f.readline().split('#time = ')[1]))
            line = f.readline()
            if '#window_position' in line:
                self.window_pos.append(float(line.split('#window_position = ')[1]))
            else:
                self.window_pos.append(0)

        df = pd.read_csv(fname, sep=',', skiprows=1)
        df.sort_values('y', inplace=True)
        self.config.append(df)
        return

    def __len__(self):
        return len(self.time)

    def get_window_pos(self):
        if np.any(np.isnan(self.window_pos)):
            warnings.warn('No window position header found.')
        else:
            return self.window_pos


class SimData:
    """This class holds output data from a single simulation.

    """

    def __init__(self, script, data_dir, threshold=0.1):

        self.data_dir = ioutil.pathize(data_dir)
        self.script = script
        self.table = pd.read_csv((self.data_dir / 'table.txt').as_posix(), sep='\t')
        self.threshold = threshold
        return

    def get_simulation_time(self):

        with (self.data_dir / 'log.txt').open(mode='r') as f:
            lines = f.readlines()

        for line in lines:
            if '//Total simulation time:' in line:
                return float(line.split('//Total simulation time:  ')[-1])

        raise ValueError('No time found.')

    def Axy(self):
        return self.table['ext_axy (rad/s)'].values

    def Az(self):
        return self.table['ext_az (rad/s)'].values

    def vdw(self, vdwcol=None):
        if vdwcol is None:
            for vdwcol in ['ext_exactdwvelavg (m/s)', 'ext_dwfinespeed (m/s)', 'ext_exactdwvelzc (m/s)']:
                if vdwcol in self.table:
                    return self.table[vdwcol].values
            raise ValueError('No vdw column in data.')
        else:
            return self.table[vdwcol].values

    def dww(self):

        dwwcol = 'ext_dwwidth (m)'
        if dwwcol in self.table:
            return self.table[dwwcol].values
        else:
            raise ValueError('No dww column in data.')

    def shift(self):
        return self.table['ext_dwpos (m)'].values

    def t(self):
        return self.table['# t (s)'].values

    def get_events(self):
        return statutil.get_events(self.t(), self.vdw(), self.threshold)

    def get_wall(self):
        return DomainWall(self.data_dir)

    def avg_vdw(self, t_cutoff):
        return np.mean(self.vdw()[self.t() > t_cutoff])

    def avg_dww(self, t_cutoff):
        return np.mean(self.dww()[self.t() > t_cutoff])

    def std_vdw(self, t_cutoff):
        return np.std(self.vdw()[self.t() > t_cutoff])

    def std_dww(self, t_cutoff):
        return np.std(self.dww()[self.t() > t_cutoff])

    def precession_freq(self):
        tf, vf = aps.LombScargle(self.t(), self.vdw()).autopower()
        peaks, _ = scs.find_peaks(vf, height=np.max(vf)*0.9)
        if len(peaks) > 0:
            return tf[peaks[0]]
        else:
            return np.nan

    def Bw_lower_bound(self, B, alpha):
        """If below the walker field Bw, we can estimate the lower bound of the walker field based on the integration
        time and the applied field.

        Parameters
        ----------
        B : float
            Applied field [T]
        alpha : float
            Gilbert damping parameter

        Returns
        -------
        float
            Lower bound for the walker field
        """

        return Bw(B, self.t()[-1], alpha)

    def anim(self, ax, **kwargs):

        wall = self.get_wall()

        line = ax.plot(wall.config[0]['x'], wall.config[0]['y'], color='k', linestyle='-')[0]
        tag = ax.text(kwargs.get('textx', 0.1), kwargs.get('texty', 0.1), '', transform=ax.transAxes)

        def init():
            line.set_xdata([])
            line.set_ydata([])
            tag.set_text('')
            return line, tag

        def anim(i):
            line.set_xdata(wall.config[i]['x'])
            line.set_ydata(wall.config[i]['y'])
            tag.set_text(f'Time: {wall.time[i]:3.3e}')
            return line, tag

        return animation.FuncAnimation(ax.get_figure(),
                                       func=anim,
                                       init_func=init,
                                       frames=len(wall)-2,
                                       interval=kwargs.get('interval', 100),
                                       blit=True)

    def anim_burst(self, ax, cmap, **kwargs):

        wall = self.get_wall()

        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)

        tag = ax.text(kwargs.get('textx', 0.1), kwargs.get('texty', 0.1), '', transform=ax.transAxes)

        def init():
            ax.plot(wall.config[0]['x'], wall.config[0]['y'], color='k', linestyle='-')
            ax.plot(wall.config[0]['x'], wall.config[0]['y'], color='k', linestyle='-')
            tag.set_text('')
            return

        def anim(i):
            ax.plot(wall.config[i+1]['x'], wall.config[i+1]['y'], color='k', linestyle='-')
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


class SimRun:
    """Simulations are run in batches. This class holds a set of simulation outputs as SimData objects.

    """

    def __init__(self, root):

        self.root = pathlib.Path(root)

        if (self.root / 'slurm_map.csv').is_file():
            self.metadata = pd.read_csv((self.root / 'slurm_map.csv').as_posix(), sep=',')
            scripts = [(self.root / script).as_posix() for script in self.metadata['script'].values]
            self.metadata['script'] = scripts

        else:
            self.metadata = get_metadata(self.root)
        self.simulations = self._get_simulations()

        return

    def _get_simulations(self):

        _s = []
        for _, row in self.metadata.iterrows():
            script = self.root / row['script']
            _s.append(SimData(script=script, data_dir=self.root / f'{script.stem}.out'))

        return _s

    def get_simulation_times(self):
        return [sim.get_simulation_time() for sim in self.simulations]

    def __getitem__(self, i):
        return self.simulations[i]

    def get_events(self):
        return [sim.get_events() for sim in self.simulations]

    def get_durations(self):
        return np.array([event.duration for event in self.get_events()])

    def get_sizes(self):
        return np.array([event.size for event in self.get_events()])

    def __repr__(self):
        return repr(self.metadata)

    def append_metadata(self, name, search_value):
        """Search through the input scripts for search_value, which is assumed to be a float. Store the found value
        for each script in self.metadata[name].
        """

        values = []
        for _, row in self.metadata.iterrows():
            values.append(find_in_script(row['script'], search_value))

        self.metadata[name] = values
        return

    def avg_vdws(self, t_cutoff=0):
        return [sim.avg_vdw(t_cutoff=t_cutoff) for sim in self.simulations]

    def avg_dwws(self, t_cutoff=0):
        return [sim.avg_dww(t_cutoff=t_cutoff) for sim in self.simulations]


def get_metadata(root):

    root = ioutil.pathize(root)

    data = {}
    for item in root.iterdir():
        script = root / (item.stem + '.mx3')
        if item.is_dir() and script.exists():
            check_dict_add_val(data, 'script', script.as_posix())

    return pd.DataFrame(data)


def check_dict_add_val(data, key, value):
    if key in data:
        data[key].append(value)
    else:
        data[key] = [value]
    return


def find_in_script(script, key):

    script = ioutil.pathize(script)

    with script.open('r') as f:
        lines = f.readlines()

    for line in lines:
        if key in line:
            return float(line.split(sep=key)[-1].split(sep=' ')[0])

    raise ValueError(f'Key {key} not found in script {script}')


def Bw(B, T, alpha):
    """When below the walker field, the magnetization will precess. Estimate the walker field given some integration
    time and applied field, assuming the period of precession is exactly the length of time you spent integrating.
    This gives a lower bound on the walker field.

    Parameters
    ----------
    B : float
        Applied fiel
    T : float
        Integration time (precession frequency)
    alpha : float
        Gilbert damping parameter

    Returns
    -------
    float
        [description]
    """

    return np.sqrt(B**2 - ((2*scc.pi*(1+alpha**2))/(scc.physical_constants['electron gyromag. ratio'][0]*T))**2)
