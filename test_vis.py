import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.animation as animation
import mx3tools.ovftools as ovftools
import mx3tools.statutil as statutil
import mx3tools.datautil as datautil
import mx3tools.plotutil as plotutil


data = datautil.OommfSim('/home/pdmurray/Desktop/Workspace/audun1ddw/micromagnetics/oommf/restoring_oommf_nx=64')