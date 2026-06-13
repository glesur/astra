#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:42:19 2021

@author: lesurg
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../pytools/"))
from vtk_io import readVTK
import inifix


parser = argparse.ArgumentParser()
parser.add_argument("-noplot",
                    default=False,
                    help="disable plotting",
                    action="store_true")
parser.add_argument("-i", "--input",
                    help="astra input file")


args, unknown=parser.parse_known_args()

conf=inifix.load(args.input)
Vini=readVTK("vtk/data.0000.vtk")
Vfin=readVTK("vtk/data.0001.vtk")

dir=conf["Physics"]["direction"]
scheme=conf["TimeIntegrator"]["method"]

error=np.sqrt(np.mean((Vini.data["rho"]-Vfin.data["rho"])**2)/np.mean((Vini.data["rho"]+Vfin.data["rho"])**2))


if(not args.noplot):
  # Comparison with exact solution
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  plt.rc('font', size=16)
  plt.close('all')

  v0=0
  v1=0
  x=0
  if dir == 0:
    v0 = Vini.data["rho"][:,1,1]
    v1 = Vfin.data["rho"][:,1,1]
    x=Vini.x
  elif dir == 1:
    v0 = Vini.data["rho"][0,:,0]
    v1 = Vfin.data["rho"][0,:,0]
    x=Vini.y
  else:
    v0 = Vini.data["rho"][0,0,:]
    v1 = Vfin.data["rho"][0,0,:]
    x=Vini.z

  plt.figure(1)
  plt.plot(x,v0)
  plt.plot(x,v1)

  plt.figure(2)
  plt.plot(x,v1-v0)



  plt.ioff()
  plt.show()

print("Error=",error)


error_threshold=0
if scheme == "rk3":
  error_threshold=1.7e-3
elif scheme == "rk2":
  error_threshold=4.0e-2
elif scheme == "euler":
  error_threshold=0.2
if(error<error_threshold):
  print("SUCCESS")
  sys.exit(0)
else:
  print("FAILED")
  sys.exit(1)
