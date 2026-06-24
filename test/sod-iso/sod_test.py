from astrapy import *
import numpy as np
#import matplotlib.pyplot as plt


def init_flow(grid, field):
  x,y,z = np.meshgrid(grid.x[IDIR], grid.x[JDIR], grid.x[KDIR], indexing='ij')
  field["rho"][:,:,:] = np.where(x<0.5, 1.0, 0.125)
  field["px1"][:,:,:] = 0
  field["px2"][:,:,:] = 0
  field["px3"][:,:,:] = 0
