import pathlib
import warnings
import subprocess
import numpy as np


class ParTree:
    """Tree structure for finding all input parameter combinations from a dictionary of input parameter ranges, i.e.,
    for a list of parameters:

        tree = ParTree({'a': [1, 2, 3, 4, 5], 'b':[2, 3]})

    The output of tree.traverse() is a list of dictionaries containing all permutations of input parameters:

        [{'b': 2, 'a': 1},
        {'b': 3, 'a': 1},
        {'b': 2, 'a': 2},
        {'b': 3, 'a': 2},
        {'b': 2, 'a': 3},
        {'b': 3, 'a': 3},
        {'b': 2, 'a': 4},
        {'b': 3, 'a': 4},
        {'b': 2, 'a': 5},
        {'b': 3, 'a': 5}]

    This is used for generating input dictionaries for Sim instances.

    """

    def __init__(self, dct, key=None, value=None):

        self.key = key
        self.value = value

        if len(dct) > 0:
            child_key = list(dct.keys())[0]
            child_values = dct[child_key]
            child_subdict = {k: v for k, v in dct.items() if k != child_key}

            self.children = [ParTree(child_subdict, child_key, child_values[i]) for i in range(len(child_values))]
        else:
            self.children = []

        return

    def which_keys(self):

        if len(self.children) == 0:
            return [self.key]
        elif self.key is not None:
            return [self.key] + self.children[0].which_keys()
        else:
            return self.children[0].which_keys()

    def traverse(self):

        if self.is_leaf():
            return [{self.key: self.value}]
        if self.is_root():
            ret = []
            for child in self.children:
                ret += child.traverse()
            return ret

        else:
            ret = []
            for child in self.children:
                dictionaries = child.traverse()
                for d in dictionaries:
                    ret.append(dict(d, **{self.key: self.value}))
            return ret

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.key is None and self.value is None


def permutations(dct):
    """Convenience function for finding all permutations of a dictionary; see ParTree class docstring

    Parameters
    ----------
    dct : dict
        Input parameter ranges. For example,

            {'a': [1,2,3],
             'b': [0.5, 0.6]}

        will return

            [{'b': 0.5, 'a': 1},
             {'b': 0.6, 'a': 1},
             {'b': 0.5, 'a': 2},
             {'b': 0.6, 'a': 2},
             {'b': 0.5, 'a': 3},
             {'b': 0.6, 'a': 3}]

    Returns
    -------
    list
        List of dictionaries containing all permutations of input parameter ranges
    """

    tree = ParTree(dct)
    return tree.traverse()


def broadcast(dct):
    """Convenience function for broadcasting a dictionary of lists.

    Parameters
    ----------
    dct : dict
        Input parameter ranges. All parameter ranges must have only 1 or N values.

    Returns
    -------
    list
        List of N dictionaries; the input parameters with N values are 'zipped' together

    """

    N = 1
    for k, v in dct.items():
        if len(v) > N:
            N = len(v)

    ret = []
    for i in range(N):
        entry = {}
        for k, v in dct.items():
            if len(v) != N:
                entry[k] = v[0]
            else:
                entry[k] = v[i]
        ret.append(entry)

    return ret


def pathize(path):
    """Takes a string or pathlib.Path object and return the corresponding pathlib.Path object.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path

    Returns
    -------
    pathlib.Path
        Returns a pathlib.Path object
    """

    if isinstance(path, str):
        return pathlib.Path(path)
    elif isinstance(path, pathlib.PurePath):
        return path
    else:
        raise TypeError(f'Invalid path type: {type(path)}')


def safely_write(path, lines, overwrite=False):
    """Safely writes a list of lines to the path. Throws an IOError if a path exists and overwrite is set to False.

    Parameters
    ----------
    path : pathlib.Path
        Path of file to write
    lines : list
        List of lines to write
    overwrite : bool, optional
        Overwrites an existing file if True (the default is False)
    """

    path = pathize(path)
    if path.exists() and not overwrite:
        raise IOError(f'{path} already exists!')
    else:
        if path.exists() and overwrite:
            warnings.warn(f'Overwriting {path}')

        with path.open(mode='w') as f:
            for line in lines:
                f.write(line)

    return


def safely_read(path):
    """Safely read a list of lines from the path.

    Parameters
    ----------
    path : pathlib.Path
        Path of file to read

    Returns
    -------
    list
        Return a list of lines from the path
    """

    path = pathize(path)

    with path.open(mode='r') as f:
        lines = f.readlines()
    return lines


def beep(sound='./coin_song.mp3'):
    """Make a beep.

    Parameters
    ----------
    sound : str, optional
        (Posix) path to the sound you want to play.

    """

    subprocess.run(['mpg123', sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return


def rmdir(d):
    d = pathlib.Path(d)
    for item in d.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    d.rmdir()
    return
