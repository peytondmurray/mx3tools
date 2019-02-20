import os
import pathlib
import subprocess
import tqdm
import pandas as pd
import json
import time
from . import ioutil


class Sim:

    def __init__(self,
                 base_script,
                 beep=False,
                 expand_vtk=False,
                 callbacks=None,
                 config=None,
                 parameters=None,
                 replace=False):

        self.config = config
        self.base_script = ioutil.pathize(base_script)
        self.lines = ioutil.safely_read(self.base_script)
        self.beep = beep
        self.callbacks = callbacks
        self.expand_vtk = expand_vtk
        self.script = pathlib.Path()
        self.outdir = pathlib.Path()
        self.suffix = ''
        self.replace = replace
        self.parameters = parameters

        return

    def ovf_to_vtk(self):
        subprocess.run([self.config['mumax_convert'], '-vtk', 'binary', (self.outdir/'*.ovf').as_posix()], shell=False)
        return

    def run(self):

        # Write the script to run
        for key, value in self.parameters.items():
            setter = getattr(self, f'set_{key}')
            setter(value)

        self.script = pathlib.Path(self.base_script.stem + self.suffix + '_' + self.base_script.suffix)
        self.outdir = pathlib.Path(self.script.stem + '.out')
        if self.outdir.exists():
            if self.replace:
                ioutil.rmdir(self.outdir)
            else:
                raise IOError('Outdir already exists!')

        # Call mumax
        ioutil.safely_write(self.script, self.lines, overwrite=self.replace)
        start_time = time.time()
        subprocess.run([self.config['mumax'], self.script], shell=False)
        print(f'Simulation time: {time.time()-start_time}')

        if self.expand_vtk:
            self.ovf_to_vtk()

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self.outdir)

        if self.beep:
            ioutil.beep()

        return

    def set_t(self, val):
        self.setval('sim_time := 10e-9', f'sim_time := {val}')
        self.suffix += f'_t={val}'
        return

    def set_r(self, val):
        self.setval('r := 0.03', f'r := {val}')
        self.suffix += f'_r={val}'
        return

    def set_size(self, val):
        self.setval('size := 128', f'size := {val}')
        self.suffix += f'_size={val}'
        return

    def set_Dbulk(self, val):
        self.setval('Dbulk = 0.0', f'Dbulk = {val}')
        self.suffix += f'_D={val}'
        return

    def set_grid(self, vals):
        self.setval('setGridSize(size, 4*size, 1)', f'setGridSize({vals[0]}, {vals[1]}, {vals[2]})')
        self.suffix = f'_size=[{vals[0]}, {vals[1]}, {vals[2]}]'
        return

    def set_bext(self, val):
        self.setval('B := 30e-3', f'B := {val}')
        self.suffix += f'_B={val}'
        return

    def setval(self, old: str, new: str):
        """Replace first instance of old string with new if it exists in lines. Return a copy of the altered set of
        lines. Throws error if the old string cannot be found.

        Parameters
        ----------
        old : str
            Old string to be replaced
        new : str
            New string to act as replacement
        """

        for i in range(len(self.lines)):
            if old in self.lines[i]:
                self.lines[i] = self.lines[i].replace(old, new, 1)
                return

        raise ValueError(f'No instances of {old} found.')

    def print_script(self):
        print(''.join(self.lines))
        return

    def get_table(self):
        return pd.read_csv((self.outdir / 'table.txt').as_posix(), sep='\t')


class Overseer:
    """Oversee a series of simulations exploring parameter space.
    """

    def __init__(self,
                 parameter_space,
                 base_script,
                 beep=True,
                 expand_vtk=True,
                 callbacks=None,
                 config='./config.json',
                 replace=False):

        self.config = self.load_config(config)
        self.tables = []
        self.simulations = self.generate_sims(parameter_space,
                                              base_script,
                                              beep=beep,
                                              expand_vtk=expand_vtk,
                                              callbacks=callbacks,
                                              config=self.config,
                                              replace=replace)

        return

    def generate_sims(self, parameter_space, base_script, beep, expand_vtk, callbacks, config, replace):

        simulations = []
        for parameters in ioutil.permutations(parameter_space):
            sim = Sim(base_script=base_script,
                      beep=beep,
                      expand_vtk=expand_vtk,
                      callbacks=callbacks,
                      parameters=parameters,
                      config=config,
                      replace=replace)
            simulations.append(sim)

        return simulations

    def run(self):
        # for sim in self.simulations:
        for sim in tqdm.tqdm(self.simulations):
            sim.run()
            self.tables.append(sim.get_table())
        return

    def data_directories(self):
        """Return a list of the data directories from all the simulations.

        Returns
        -------
        list
            Output directories
        """

        return [simulation.outdir for simulation in self.simulations]

    def tabulate(self):
        """Tables are automatically gathered upon completing each simulation, so this should only need to be called
        if you want to re-tabulate the data.
        """

        self.tables = []

        for sim in tqdm.tqdm(self.simulations):
            self.tables.append(pd.read_csv(sim.get_table()))

        return self.tables

    def load_config(self, config='./config.json') -> dict:
        """Load a config file, which specifies global config options, e.g. the location of mumax, etc

        Parameters
        ----------
        config : str, dict, pathlib.Path

        Returns
        -------
        dict
            Global config options, returned as a dict
        """

        if isinstance(config, dict):
            return config
        elif isinstance(config, pathlib.Path):
            config = config.as_posix()
        elif isinstance(config, str):
            pass
        else:
            raise ValueError(f'Unsupported config file type: {type(config)}')

        with open(config, 'r') as f:
            config = json.load(f)
        return config
