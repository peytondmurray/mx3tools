# Data structures related to simulation output files

import re
import numpy as np
import pandas as pd
import pathlib
from . import statutil
from . import ioutil


class SimData:
    """This class holds output data from a single simulation.

    """

    def __init__(self, script, data_dir, threshold=0.1):

        self.data_dir = data_dir
        self.script = script
        self.table = pd.read_csv((data_dir / 'table.txt').as_posix(), sep='\t')
        self.threshold = threshold
        return

    def get_simulation_time(self):

        with (self.data_dir / 'log.txt').open(mode='r') as f:
            lines = f.readlines()

        for line in lines:
            if '//Total simulation time:' in line:
                return float(line.split('//Total simulation time:  ')[-1])

        raise ValueError('No time found.')

    def get_Axy(self):
        return self.table['ext_axy (rad/s)'].values

    def get_Az(self):
        return self.table['ext_az (rad/s)'].values

    def get_vdw(self, vdwcol='ext_exactdwvelavg (m/s)'):
        return self.table[vdwcol].values

    def get_t(self):
        return self.table['# t (s)'].values

    def get_events(self, vdwcol='ext_exactdwvelavg (m/s)'):
        return statutil.get_events(self.get_t(), self.get_vdw(vdwcol), self.threshold)

    def get_dwconfigs(self):
        dwconfigs = []
        for item in self.data_dir.iterdir():
            if re.search(r'domainwall\d{6}.csv', item.name) is not None:
                dwconfigs.append(item.name)

        return [pd.read_csv((self.data_dir / item), sep=',', skiprows=1) for item in sorted(dwconfigs)]


class SimRun:
    """Simulations are run in batches. This class holds a set of simulation outputs as SimData objects.

    """

    def __init__(self, root):

        self.root = pathlib.Path(root)

        if (self.root / 'slurm_map.csv').is_file():
            self.metadata = pd.read_csv((self.root / 'slurm_map.csv').as_posix(), sep=',')
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
