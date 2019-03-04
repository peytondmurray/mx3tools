import numpy as np
import numba as nb
import tqdm
from . import ovftools


@nb.jit(nopython=True)
def phi(m: np.ndarray) -> np.ndarray:
    return phi_t(m.reshape(1, *m.shape))[0]


@nb.jit(nopython=True)
def theta(m: np.ndarray) -> np.ndarray:
    return theta_t(m.reshape(1, *m.shape))[0]


@nb.jit(nopython=True)
def phi_t(m_t: np.ndarray) -> np.ndarray:
    """Azimuthal angle. See theta() docstring for more info.

    Parameters
    ----------
    m_t : np.ndarray

    Returns
    -------
    np.ndarray
    """

    return np.arctan2(m_t[:, :, :, :, 1], m_t[:, :, :, :, 0])


@nb.jit(nopython=True)
def theta_t(m_t: np.ndarray) -> np.ndarray:
    """Polar angle.

    Parameters
    ----------
    m_t : np.ndarray
        Magnetization vectors. This array is stored in zyx coordinate format as 3-tuples of (x, y, z) vector components.
        i.e.,

            m[i, j, k, l, m]
              |  |  |  |  |
              |  |  |  |  ---- Denotes the vector component 0 = x, 1 = y, 2 = z
              |  |  |  ------- x cartesian coordinate
              |  |  ---------- y cartesian coordinate
              |  ------------- z cartesian coordinate
              ---------------- timestep

    Returns
    -------
    np.ndarray
        Theta polar angle array, with same shape as m
    """

    return np.arccos(m_t[:, :, :, :, 2])


@nb.jit
def mask_dw(m: np.ndarray, band_size: int=10) -> np.ndarray:
    return mask_dw_t(m.reshape(1, *m.shape), band_size=band_size)[0]


@nb.jit
def mask_dw_t(m_t: np.ndarray, band_size: int=10) -> np.ndarray:
    """Mask the magnetization array with zeros, except in the region surrounding the domain wall, which are filled with
    ones. DW is assumed to run along the y-direction.

    Parameters
    ----------
    m_t : np.ndarray
        Magnetization vector array as a function of time; shape (Nt, Nz, Ny, Nx, 3). First coordinate denotes timestep,
        with other dimensions as in theta() docstring.
    band_size : int, optional
        Size of integration region across the domain wall, with band_size number of cells on either size of the wall
        being included in the calculation.

    Returns
    -------
    np.ndarray
        [description]
    """

    i_center = np.empty((m_t.shape[0], m_t.shape[1], m_t.shape[2]))
    for i in range(m_t.shape[0]):
        i_center[i, :, :] = dw_center(m_t[i])

    ret = np.zeros(m_t.shape)
    for i in range(m_t.shape[0]):
        for j in range(m_t.shape[1]):
            for k in range(m_t.shape[2]):
                ret[i, j, k, i_center[i, j, k]-band_size:i_center[i, j, k]+band_size+1, :] = 1
    return ret


@nb.jit(nopython=True)
def dw_center(m: np.ndarray) -> np.ndarray:
    """Return the index of the center of the domain wall across the x-direction; domain wall is assumed to run along y.

    Parameters
    ----------
    m : np.ndarray
        Magnetization array is assumed to be of shape (Nz, Ny, Nx, 3). Domain wall is considered to be centered upon the
        zero-crossing of the Mz component. See theta() docstring for more info.

    Returns
    -------
    np.ndarray
        Array of shape (Nz, Ny). i_center[i, :] holds the x-index of domain wall center.
    """

    i_center = np.empty((m.shape[0], m.shape[1]))
    for i in range(m.shape[0]):
        # i_center[i, :] = np.argmin(np.abs(m[i, :, :, 2]), axis=1)
        i_center[i, :] = argmin_2d_along_x(np.abs(m[i, :, :, 2]))

    return i_center


@nb.jit(nopython=True)
def mag_xy(m: np.ndarray) -> np.ndarray:
    return mag_xy_t(m.reshape(1, *m.shape))[0]


@nb.jit(nopython=True)
def mag_xy_t(m_t: np.ndarray) -> np.ndarray:
    """Return the magitude of the magnetization in the xy plane.

    Parameters
    ----------
    m_t : np.ndarray
        Magnetization vector array as a function of time; shape (Nt, Nz, Ny, Nx, 3). First coordinate denotes timestep,
        with other dimensions as in theta() docstring.

    Returns
    -------
    np.ndarray
        [description]
    """

    return np.sqrt(m_t[:, :, :, :, 0]**2 + m_t[:, :, :, :, 1]**2)


@nb.jit(nopython=True)
def argmin_2d_along_x(arr):

    ret = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        ret[i] = np.argmin(arr[i])

    return ret
