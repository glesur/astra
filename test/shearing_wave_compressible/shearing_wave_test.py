from astrapy import *
import numpy as np
#import matplotlib.pyplot as plt

# The output function
# the only argument is dataBlockHost python object, wrapping a dataBlockHost Idefix object
def output(grid, field, t, n):
  # only process #0 performs the output
  shear = 1.5
  if prank==0:
    mode = "a"
    if n==0:
      mode = "w"
    f = open("mode_amplitude.dat",mode)
    kx = int(t*shear+0.5)
    if n==0:
      f.write("t  vx   vy   vz\n")

    f.write("%e   %e   %e   %e\n" % (t, np.real(field["px1"][kx,1,4]), np.real(field["px2"][kx,1,4]), np.real(field["px3"][kx,1,4])))
    f.close()



def init_flow(grid, field):
  if prank==0:
    field["px1"][0,1,4] = 1.0e-2*grid.npr_glob[IDIR]*grid.npr_glob[JDIR]*grid.npr_glob[KDIR]
    field["rho"][0,0,0] = grid.npr_glob[IDIR]*grid.npr_glob[JDIR]*grid.npr_glob[KDIR]
  # Field amplitude
  #field["vx1"] = 0*x
