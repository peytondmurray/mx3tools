# This code is based on oommfdecode.py by Duncan Parkes:
# https://github.com/deparkes/OOMMFTools/blob/master/oommftools/core/oommfdecode.py
# The _binary_decode function is taken almost directly from there, except the order in which the OVF data is stored
# has been changed to conform with numpy's array indexing conventions.
#
# The _fast_binary_decode function uses numpy's ndarray constructor to eliminate the need for loops, dramatically
# reducing the time needed to move the data read from the file object into an array (~100x speedup).

import re
import numpy as np
import struct
import pathlib
import tqdm
from . import ioutil
from . import util





def unpack_slow(path):
    path = ioutil.pathize(path)

    with path.open('rb') as f:
        headers = _read_header(f)

        if headers['data_type'][3] == 'Text':
            return _text_decode(f, headers)

        elif headers['data_type'][3] == 'Binary':
            chunk_size = int(headers['data_type'][4])
            endianness = _endianness(f, chunk_size)
            decoder = _byte_decoder(endianness)
            return _binary_decode(f, chunk_size, decoder, headers, endianness)


def unpack(path):
    path = ioutil.pathize(path)

    with path.open('rb') as f:
        headers = _read_header(f)

        if headers['data_type'][3] == 'Text':
            return _text_decode(f, headers)

        elif headers['data_type'][3] == 'Binary':
            chunk_size = int(headers['data_type'][4])
            return _fast_binary_decode(f, chunk_size, headers, _endianness(f, chunk_size))


def _read_header(fobj):
    """Read headers from OVF file object. fobj must be opened in 'rb' mode (read as bytes).

    Parameters
    ----------
    fobj : file
        OVF file to read, must be opened in bytes mode (mode='rb')

    Returns
    -------
    dict
        Dictionary containing the [important] header keys and values
    """

    headers = {'SimTime': -1, 'Iteration': -1, 'Stage': -1, 'MIFSource': ''}

    line = ''
    while 'Begin: Data' not in line:

        line = fobj.readline().strip().decode()

        for key in ["xbase",
                    "ybase",
                    "zbase",
                    "xstepsize",
                    "ystepsize",
                    "zstepsize",
                    "xnodes",
                    "ynodes",
                    "znodes",
                    "valuemultiplier"]:
            if key in line:
                headers[key] = float(line.split(': ')[1])

        if 'Total simulation time' in line:
            headers['SimTime'] = float(line.split(':')[-1].strip().split()[0].strip())
        elif 'Iteration' in line:
            headers['Iteration'] = float(line.split(':')[2].split(',')[0].strip())
        # elif 'Stage' in line:
        #     headers['Stage'] = float(line.split(':')[2].split(',')[0].strip())
        elif 'MIF source file' in line:
            headers['MIFSource'] = line.split(':', 2)[2].strip()
        else:
            continue

    headers['data_type'] = line.split()

    return headers


def _byte_decoder(endianness):
    return struct.Struct(endianness)


def _endianness(f, nbytes):
    buffer = f.read(nbytes)

    big_endian = {4: '>f', 8: '>d'}
    little_endian = {4: '<f', 8: '<d'}
    value = {4: 1234567.0, 8: 123456789012345.0}

    if struct.unpack(big_endian[nbytes], buffer)[0] == value[nbytes]:       # Big endian?
        return big_endian[nbytes]
    elif struct.unpack(little_endian[nbytes], buffer)[0] == value[nbytes]:  # Little endian?
        return little_endian[nbytes]
    else:
        raise IOError(f'Cannot decode {nbytes}-byte order mark: ' + hex(buffer))


def _binary_decode(f, chunk_size, decoder, headers, dtype):

    data = np.empty((int(headers['znodes']),
                     int(headers['ynodes']),
                     int(headers['xnodes']), 3), dtype=dtype)

    for k in range(data.shape[0]):
        for j in range(data.shape[1]):
            for i in range(data.shape[2]):
                for coord in range(3):
                    data[k, j, i, coord] = decoder.unpack(f.read(chunk_size))[0]

    return data*headers.get('valuemultiplier', 1)


def _text_decode(f, headers):

    data = np.empty((int(headers['znodes']),
                     int(headers['ynodes']),
                     int(headers['xnodes']), 3), dtype=float)

    for k in range(data.shape[0]):
        for j in range(data.shape[1]):
            for i in range(data.shape[2]):
                text = f.readline().strip().split()
                data[k, j, i] = (float(text[0]), float(text[1]), float(text[2]))

    return data*headers.get('valuemultiplier', 1)


def _fast_binary_decode(f, chunk_size, headers, dtype):

    xs, ys, zs = (int(headers['xnodes']), int(headers['ynodes']), int(headers['znodes']))
    ret = np.ndarray(shape=(xs*ys*zs, 3),
                     dtype=dtype,
                     buffer=f.read(xs*ys*zs*3*chunk_size),
                     offset=0,
                     strides=(3*chunk_size, chunk_size))

    return ret.reshape((zs, ys, xs, 3))


def _fast_binary_decode_scalars(f, chunk_size, headers, dtype):

    xs, ys, zs = (int(headers['xnodes']), int(headers['ynodes']), int(headers['znodes']))
    ret = np.ndarray(shape=(xs*ys*zs, 1),
                     dtype=dtype,
                     buffer=f.read(xs*ys*zs*chunk_size),
                     offset=0,
                     strides=(chunk_size, chunk_size))

    return ret.reshape((zs, ys, xs))


def group_unpack(path, pattern='m'):
    """Unpack a bunch of .ovf files to a numpy array.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of files to unpack. If path is a *.ovf, the directory is searched for other similarly named files. If
        it is a directory ending in .out, the directory is searched for files matching {pattern}*.ovf.
    pattern : str, optional
        File search pattern; finds all files in path matching {pattern}[0-9]+.ovf. By default this is m.

    Returns
    -------
    np.ndarray
        Array of floats extracted from the .ovf files.
    """

    path = ioutil.pathize(path)

    if path.suffix == '.out' or path.is_dir():
        files = sorted(path.glob(f'{pattern}*.ovf'))
        if len(files) == 0:
            files = sorted(path.glob(f'{pattern}*.omf'))

        if len(files) == 0:
            raise ValueError(f'No .omf or .ovf files found in {path}')

    elif path.suffix == '.ovf':
        pattern = re.search('\D+', path.stem)[0]
        files = sorted(path.parent.glob(f'{pattern}*.ovf'))
    elif path.suffix == '.omf':
        pattern = re.search('.+', path.stem)[0]
        files = sorted(path.parent.glob(f'{pattern}*.omf'))
    else:
        raise ValueError(f'Invalid path: {path} must end in .out, .ovf, or .omf')

    if len(files) == 0:
        raise ValueError(f'No .ovf files found in {path}')

    return np.array([unpack(f) for f in tqdm.tqdm(files, desc='Unpacking .ovf files')])


def unpack_scalars(path):
    path = ioutil.pathize(path)

    with path.open('rb') as f:
        headers = _read_header(f)

        if headers['data_type'][3] == 'Binary':
            chunk_size = int(headers['data_type'][4])
            return _fast_binary_decode_scalars(f, chunk_size, headers, _endianness(f, chunk_size))

        else:
            raise NotImplementedError


def as_rodrigues(path, fname):
    """For each m*.ovf file in the given directory, generate a corresponding .csv containing the indices, rotation
    axes, and angles needed to map a uniform[0, 0, 1] magnetization state to the data.

    Parameters
    ----------
    path: [type]
        [description]
    fname: [type]
        [description]
    """

    path = ioutil.pathize(path)
    data = group_unpack(path)

    for i, item in enumerate(data):
        write_rodrigues(f'{fname}_rodrigues_{i}', item)

    return


def write_rodrigues(fname, data):
    """Given an input set of magnetization data, write an output csv file containing the rotation axis and angle
    needed to map a uniform[0, 0, 1] magnetization to the data.

    Parameters
    ----------
    fname: str
        File name to write
    data: np.ndarray
        Array containing the magnetization vectors; vectors are stored as (mx, my, mz) 3-tuples, where the magnetization
        at site[ix, iy, iz] is given by data[iz, iy, ix].
    """

    with open(fname, 'w') as f:
        f.write('xi,yi,zi,kx,ky,kz,theta')

        reference = np.array([0, 0, 1])

        for iz in range(data.shape[0]):
            for iy in range(data.shape[1]):
                for ix in range(data.shape[2]):
                    k = np.cross(data[iz, iy, ix], reference)
                    theta = np.arcsin(np.dot(k, reference)/(np.abs(k)))
                    f.write(f'{ix},{iy},{iz},{k[0]},{k[1]},{k[2]},{theta}')

    return


# def write_successive_rodrigues(path, fname):

#     path = ioutil.pathize(path)
#     data = group_unpack(path)

#     for i in range(len(data)):
