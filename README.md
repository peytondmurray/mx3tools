# mx3tools
This is a collection of python tools I made for running and analyzing simulations using Mumax3. It's not intended as a general use package, though it does have a number of utilities that would be useful to anyone running micromagnetic simulations.

## Installation
```python
cd mx3tools
pip install .
```

If you want to install mx3tools so that the source is editable, use `pip install -e .`.

## Feature Highlights
Load a set of simulation files from `simulation.out/*.ovf` into a numpy array:
```python
from mx3tools import ovftools
data = ovftools.group_unpack('simulation.out/')
```

The data will be loaded into an array of shape `(n_files, nz, ny, nx, 3)`, where `n_files` is the number of `.ovf` files in the directory, `nz`, `ny`, and `nx` are the number of simulation cells along z, y, and x. The last dimension holds the orientation of the magnetization as a vector, `[mx, my, mz]`.
