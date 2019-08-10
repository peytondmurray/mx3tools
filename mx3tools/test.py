import numpy as np
from . import util
from . import datautil
# import util
# import datautil
import pytest

@pytest.mark.skip
def test_fornberg():

    x = np.arange(-4,5)
    print(x)
    print(util.fornberg(x, 0, 4).T)

    return

def test_indexing():

    data = datautil.SimRun('/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.0e-3/2019-06-24/')

    df1 = data.metadata.iloc[0]
    print(df1)
    print(type(df1))

    return

test_indexing()

# test_fornberg()
