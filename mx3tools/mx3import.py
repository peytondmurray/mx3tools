# This code is based on oommfdecode.py by Duncan Parkes:
# https://github.com/deparkes/OOMMFTools/blob/master/oommftools/core/oommfdecode.py
# and also ovftools.py by Peyton Murray
# https://github.com/peytondmurray/mx3tools/blob/master/mx3tools/ovftools.py
#
# How to use: 
# place this file in your working folder and use "from mx3import import unpack"
# and then use unpack("mumax-generated-file.ovf")
#
# works on both text and binary format (prefer the binary! it is much less heavy)
# works on both scalar and vector data (ouput shape is always [znodes, ynodes, xnodes, valuedim] )

import numpy as np
import struct
import pathlib

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


def unpack(path):
    path = pathize(path)

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
                    "valuemultiplier",
                    "valuedim"]:
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

def _text_decode(f, headers):
    
    arrshape=[int(headers[key]) for key in ['znodes','ynodes','xnodes','valuedim']]
    
    data = np.loadtxt(f, max_rows=np.prod(arrshape[:3])).reshape(arrshape)

    return data*headers.get('valuemultiplier', 1)


def _fast_binary_decode(f, chunk_size, headers, dtype):

    arrshape=[int(headers[key]) for key in ['znodes','ynodes','xnodes','valuedim']]
    ret = np.ndarray(shape=arrshape,
                     dtype=dtype,
                     buffer=f.read(np.prod(arrshape)*chunk_size),
                     offset=0
                     )

    return ret
