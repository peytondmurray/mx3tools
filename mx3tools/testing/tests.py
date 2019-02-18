import numpy as np
import pytest
import matplotlib.pyplot as plt
import mx3tools.ovftools as ovftools


def test_fast_ovf_read():
    data = ovftools.unpack_slow('/home/pdmurray/Projects/dmidw/dmidw.out/m000000.ovf')
    slow_data = ovftools.unpack('/home/pdmurray/Projects/dmidw/dmidw.out/m000000.ovf')
    assert np.array_equal(data, slow_data)
    return


def plot_ovf_readers():
    data = ovftools.unpack('/home/pdmurray/Projects/dmidw/dmidw.out/m000000.ovf')
    slow_data = ovftools.unpack_slow('/home/pdmurray/Projects/dmidw/dmidw.out/m000000.ovf')

    comp = 2

    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.imshow(data[0, :, :, comp], origin='lower')
    ax.set_title('fast_data')
    ax = fig.add_subplot(132)
    ax.imshow(slow_data[0, :, :, comp], origin='lower')
    ax.set_title('slow data')
    ax = fig.add_subplot(133)
    ax.imshow(np.equal(data[0, :, :, comp], slow_data[0, :, :, comp]).astype(int), cmap='Reds', origin='lower')
    ax.set_title('np.equal(fast, slow)')
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    test_fast_ovf_read()
