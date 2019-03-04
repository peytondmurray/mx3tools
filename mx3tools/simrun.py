import os
import pathlib
import subprocess
import tqdm
import pandas as pd
import json_tricks as json
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
                 replace=False,
                 slurm_base=None,
                 script_override=None):

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
        self.slurm_base = slurm_base
        self.script_override = script_override

        self.set_parameters()
        self.generate_script()

        return

    def ovf_to_vtk(self):
        subprocess.run([self.config['mumax_convert'], '-vtk', 'binary', (self.outdir/'*.ovf').as_posix()], shell=False)
        return

    def run(self):

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

    def generate_script(self):

        if self.script_override is None:
            self.script = pathlib.Path(self.base_script.stem + self.suffix + '_' + self.base_script.suffix)
        else:
            self.script = pathlib.Path(self.script_override)

        self.outdir = pathlib.Path(self.script.stem + '.out')

        if self.outdir.exists():
            if self.replace:
                ioutil.rmdir(self.outdir)
            else:
                raise IOError('Outdir already exists!')

        # Call mumax
        ioutil.safely_write(self.script, self.lines, overwrite=self.replace)

        return

    def set_parameters(self):
        # Write the script to run
        for key, value in self.parameters.items():
            setter = getattr(self, f'set_{key}')
            setter(value)

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
        self.setval('size := 128', f'size := {val:d}')
        self.suffix += f'_size={val:d}'
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

    def set_seed(self, val):
        self.setval('random_seed := 123', f'random_seed := {val}')
        self.suffix += f'_seed={val}'
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

    def generate_slurm(self):
        if self.slurm_base is not None:
            slurm_lines = ioutil.safely_read(self.slurm_base)
            slurm_lines = self.slurm_setval(slurm_lines, '#SBATCH -J suffix', f'#SBATCH -J {self.suffix}')
            slurm_lines = self.slurm_setval(slurm_lines, 'srun mumax3 script.mx3', f'srun mumax3 {self.script}')
            ioutil.safely_write(f'slurm{self.suffix}.sh', slurm_lines, overwrite=self.replace)

        else:
            raise ValueError('No input slurm batch script was given on instantiation.')

        return

    def slurm_setval(self, slurm_lines, old, new):
        """Replace first instance of old string with new if it exists in lines. Return a copy of the altered set of
        lines. Throws error if the old string cannot be found.

        Parameters
        ----------
        old : str
            Old string to be replaced
        new : str
            New string to act as replacement
        """

        for i in range(len(slurm_lines)):
            if old in slurm_lines[i]:
                slurm_lines[i] = slurm_lines[i].replace(old, new, 1)
                return slurm_lines

        raise ValueError(f'No instances of {old} found in slurm base.')


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
                 replace=False,
                 slurm_base=None,
                 generate_slurm_array=False):

        self.config = self.load_config(config)
        self.tables = []
        self.simulations = self.generate_sims(parameter_space,
                                              base_script,
                                              beep=beep,
                                              expand_vtk=expand_vtk,
                                              callbacks=callbacks,
                                              config=self.config,
                                              replace=replace,
                                              slurm_base=slurm_base,
                                              generate_slurm_array=generate_slurm_array)

        return

    def __getitem__(self, arg):
        return self.simulations[arg]

    def generate_sims(self,
                      parameter_space,
                      base_script,
                      beep,
                      expand_vtk,
                      callbacks,
                      config,
                      replace,
                      slurm_base,
                      generate_slurm_array):

        simulations = []

        if generate_slurm_array:

            slurm_map = {}
            for i, parameters in enumerate(ioutil.permutations(parameter_space)):
                script_name = '{}_{}.mx3'.format(ioutil.pathize(base_script).stem, i)
                simulations.append(Sim(base_script=base_script,
                                       beep=beep,
                                       expand_vtk=expand_vtk,
                                       callbacks=callbacks,
                                       parameters=parameters,
                                       config=config,
                                       replace=replace,
                                       script_override=script_name))
                slurm_map[i] = {'parameters': parameters, 'script': script_name}

            with open('slurm_map.json', 'w') as f:
                json.dump(slurm_map, f, indent=4)

        elif slurm_base is not None:

            for parameters in ioutil.permutations(parameter_space):
                simulations.append(Sim(base_script=base_script,
                                       beep=beep,
                                       expand_vtk=expand_vtk,
                                       callbacks=callbacks,
                                       parameters=parameters,
                                       config=config,
                                       replace=replace,
                                       slurm_base=slurm_base))

            self.generate_slurms()

        else:
            for parameters in ioutil.permutations(parameter_space):
                simulations.append(Sim(base_script=base_script,
                                       beep=beep,
                                       expand_vtk=expand_vtk,
                                       callbacks=callbacks,
                                       parameters=parameters,
                                       config=config,
                                       replace=replace))

        return simulations

    def generate_slurms(self):
        for sim in self.simulations:
            sim.generate_slurm()
        return

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
