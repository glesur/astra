import argparse
import os
import shutil
import subprocess
import sys
import re

import numpy as np
import matplotlib.pyplot as plt
import runpy

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class astraTest:
  def __init__ (self, buildDirInput=""):
    parser = argparse.ArgumentParser()

    parser.add_argument("-noplot",
                        dest="noplot",
                        help="disable plotting in standard tests",
                        action="store_false")

    parser.add_argument("-nopython",
                        dest="nopython",
                        help="disable Python bindings in standard tests",
                        action="store_true")

    parser.add_argument("-cmake",
                        default=[],
                        help="CMake options",
                        nargs='+')

    parser.add_argument("-buildDir",
                        default="",
                        help="Build directory")

    parser.add_argument("-cuda",
                        help="Test on Nvidia GPU using CUDA",
                        action="store_true")

    parser.add_argument("-intel",
                        help="Test compiling with Intel OneAPI",
                        action="store_true")

    parser.add_argument("-hip",
                        help="Test on AMD GPU using HIP",
                        action="store_true")

    parser.add_argument("-mpi",
                        help="Enable MPI",
                        action="store_true")

    parser.add_argument("-init",
                    help="Reinit reference files for non-regression tests (dangerous!)",
                    action="store_true")

    parser.add_argument("-Werror",
                    help="Consider warnings as errors",
                    action="store_true")


    parser.add_argument("-v", "--verbose",
                    help="Enable verbose mode, by not capturing the output.",
                    action="store_true")


    args, unknown=parser.parse_known_args()

    # transform all arguments from args into attributes of this instance
    self.__dict__.update(vars(args))

    # override build dir if none was provided
    if self.buildDir == "" and buildDirInput != "":
      self.buildDir = buildDirInput

    # store the full path of problem directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    self.astraDir = os.path.join(script_dir, "..")

    self.referenceDirectory = os.path.join(self.astraDir,"reference")


  def _genCmakeCommand(self):
    comm=["cmake"]
    # add source directory
    comm.append(self.astraDir)


    # add specific options
    for opt in self.cmake:
      comm.append("-D"+opt)

    # Enable python by default
    if not self.nopython:
      comm.append("-DAstra_PYTHON=ON")

    if self.cuda:
      comm.append("-DKokkos_ENABLE_CUDA=ON")
      # disable fmad operations on Cuda to make it compatible with CPU arithmetics
      comm.append("-DAstra_CXX_FLAGS=--fmad=false")
      # disable Async cuda malloc for tests performed on old UCX implementations
      comm.append("-DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF")

    if self.intel:
      # disable fmad operations on Cuda to make it compatible with CPU arithmetics
      comm.append("-DAstra_CXX_FLAGS=-fp-model=strict")
      comm.append("-DCMAKE_CXX_COMPILER=icpx")
      comm.append("-DCMAKE_C_COMPILER=icx")

    if self.hip:
      comm.append("-DKokkos_ENABLE_HIP=ON")
      # disable fmad operations on HIP to make it compatible with CPU arithmetics
      comm.append("-DAstra_CXX_FLAGS=-ffp-contract=off")

    if(self.Werror):
      comm.append("-DAstra_WERROR=ON")


    if(self.mpi):
      comm.append("-DAstra_MPI=ON")
    else:
      comm.append("-DAstra_MPI=OFF")

    # always do a fresh build to avoid any issue with old files
    comm.append("--fresh")
    # ok
    return comm


  def configure(self):
    # log

    # gen command
    comm = self._genCmakeCommand()

    # log
    print("***************** CALLING CMAKE *******************")
    print(f"mkdir -p {self.buildDir}")
    print(f"cd {self.buildDir}")
    print(' '.join(comm))
    print("***************************************************")

    # run
    try:
        # do cleanup
        self.clean()

        cmake=subprocess.run(comm, cwd=os.path.abspath(self.buildDir))
        cmake.check_returncode()
    except subprocess.CalledProcessError as e:
        print(bcolors.FAIL+"***************************************************")
        print("Cmake failed")
        print("***************************************************"+bcolors.ENDC)
        raise e
    finally:
        # remind for next time
        self.lastCmakeCmd = ' '.join(comm)

  def clean(self):
    # remove the build directory before re-creating it
    if os.path.exists(self.buildDir):
      shutil.rmtree(self.buildDir)

    # recreate
    os.makedirs(self.buildDir, exist_ok=False)

  def build(self,jobs=8):
    try:
        comm = ["make","-j"+str(jobs)]

        print("***************************************************")
        print(f"cd {os.getcwd()}")
        print(' '.join(comm))
        print("***************************************************")

        make=subprocess.run(comm, cwd=os.path.abspath(self.buildDir))
        make.check_returncode()
    except subprocess.CalledProcessError as e:
        print(bcolors.FAIL+"***************************************************")
        print("Compilation failed")
        print("***************************************************"+bcolors.ENDC)
        raise e

  def run(self, inputFile="", problemDir="./", np=2, nowrite=False, restart=-1):
      # log

      comm=[os.path.abspath(os.path.join(self.buildDir,"astra"))]
      if inputFile:
          comm.append("-i")
          comm.append(inputFile)
      if self.mpi:
          comm.insert(0,"mpirun")
          comm.insert(1,"-np")
          comm.insert(2,str(np))

      if nowrite:
          comm.append("-nowrite")

      if self.Werror:
        comm.append("-Werror")

      if restart>=0:
        comm.append("-restart")
        comm.append(str(restart))

      print("***************************************************")
      print(f"cd {os.getcwd()}")
      print(' '.join(comm))
      print("***************************************************")

      try:
          make=subprocess.run(comm,cwd=os.path.abspath(problemDir))
          make.check_returncode()
      except subprocess.CalledProcessError as e:
          print(bcolors.FAIL+"***************************************************")
          print("Execution failed")
          print("***************************************************"+bcolors.ENDC)
          raise e

      self._readLog(problemDir=problemDir)

  def _readLog(self,problemDir="./"):

    if not os.path.exists(os.path.join(problemDir, './astra.0.log')):
      # When no idefix file is produced, we leave
      return

    with open(os.path.join(problemDir, './astra.0.log'), 'r') as file:
      log = file.read()


    if "Kokkos CUDA target ENABLED" in log:
      self.cuda = True
    else:
      self.cuda = False

    if "Kokkos HIP target ENABLED" in log:
      self.hip = True
    else:
      self.hip = False


    self.mpi=False
    if "MPI ENABLED" in log:
      self.mpi=True


    # Get input file from log
    line = re.search('(?<=Input Parameters using input file )(.*)', log)
    self.inifile=line.group(0)[:-1]

    # Get performances from log
    line = re.search('Main: Perfs are (.*) cell', log)
    self.perf=float(line.group(1))

  def checkOnly(self, filename, tolerance=0):

    # Assumes the code has been run manually using some configuration, so we simply
    # do the test suite witout configure/compile/run
    self._readLog()
    self._showConfig()
    if self.cuda or self.hip:
      print(bcolors.WARNING+"***********************************************")
      print("WARNING: Idefix guarantees floating point arithmetic accuracy")
      print("ONLY when fmad instruction are explicitely disabled at compilation.")
      print("Otheriwse, this test will likely fail.")
      print("***********************************************"+bcolors.ENDC)
    self.standardTest()
    self.nonRegressionTest(filename, tolerance)

  def standardTest(self, problemDir="./", inputFile="astra.ini"):
    # log and in fake mode do not execute.

    if os.path.exists(os.path.join(problemDir, 'validate.py')):
      oldPwd = os.getcwd()
      os.chdir(problemDir)
      original_argv = sys.argv
      sys.argv = [sys.argv[0], "-i", inputFile]
      if self.noplot:
        sys.argv.append("-noplot")

      print(bcolors.OKCYAN+"Running standard test...")
      try:
          runpy.run_path("validate.py", init_globals=globals())
      except SystemExit as e:
        if e.code != 0:
          print(bcolors.FAIL+"***************************************************")
          print("Standard test failed with exit code ", e.code)
          print("***************************************************"+bcolors.ENDC)
          raise e
        else:
          pass  # or log e.code if you need the exit code
      print(bcolors.OKCYAN+"Standard test succeeded"+bcolors.ENDC)
      sys.argv = original_argv
      os.chdir(oldPwd)
    else:
      print(bcolors.WARNING+"No standard test validate.py for this test"+bcolors.ENDC)
    sys.stdout.flush()

  """ def nonRegressionTest(self, filename,tolerance=0):
    # log and in fake mode do not execute.

    fileref=os.path.join(self.referenceDirectory, self.testDir, self._getReferenceFilename())
    if not(os.path.exists(fileref)):
      raise Exception("Reference file "+fileref+ " doesn't exist")

    filetest=filename
    if not(os.path.exists(filetest)):
      raise Exception("Test file "+filetest+ " doesn't exist")

    Vref=readDump(fileref)
    Vtest=readDump(filetest)
    error=self._computeError(Vref,Vtest)
    if error > tolerance:
      print(bcolors.FAIL+"Non-Regression test failed!")
      self._showConfig()
      print(bcolors.ENDC)
      if self.ploterr:
        self._plotDiff(Vref,Vtest)
      assert error <= tolerance, bcolors.FAIL+"Error (%e) above tolerance (%e)"%(error,tolerance)+bcolors.ENDC
    print(bcolors.OKGREEN+"Non-regression test succeeded with error=%e"%error+bcolors.ENDC)
    sys.stdout.flush()

  def compareDump(self, file1, file2,tolerance=0):

    Vref=readDump(file1)
    Vtest=readDump(file2)
    error=self._computeError(Vref,Vtest)
    if error > tolerance:
      print(bcolors.FAIL+"Files are different !")
      print(bcolors.ENDC)

      self._plotDiff(Vref,Vtest)
      assert error <= tolerance, bcolors.FAIL+"Error (%e) above tolerance (%e)"%(error,tolerance)+bcolors.ENDC
    print(bcolors.OKGREEN+"Files are identical up to error=%e"%error+bcolors.ENDC)
    sys.stdout.flush()


  def makeReference(self,filename):

    self._readLog()
    targetDir = os.path.join(self.referenceDirectory,self.testDir)
    if not os.path.exists(targetDir):
      print("Creating reference directory")
      os.makedirs(targetDir, exist_ok=True)
    fileout = os.path.join(targetDir, self._getReferenceFilename())
    if(os.path.exists(fileout)):
      ans=input(bcolors.WARNING+"This will overwrite already existing reference file:\n"+fileout+"\nDo you confirm? (type yes to continue): "+bcolors.ENDC)
      if(ans != "yes"):
        print(bcolors.WARNING+"Reference creation aborpted"+bcolors.ENDC)
        return

    shutil.copy(filename,fileout)
    print(bcolors.OKGREEN+"Reference file "+fileout+" created"+bcolors.ENDC)
    sys.stdout.flush() """

  def _showConfig(self):
    print("**************************************************************")
    if self.cuda:
      print("Nvidia Cuda enabled.")
    if self.hip:
      print("AMD HIP enabled.")
    print("CMake Opts: " +" ".join(self.cmake))
    print("Definitions file:"+self.definitions)
    print("Input File: "+self.inifile)
    if(self.single):
      print("Precision: Single")
    else:
      print("Precision: Double")
    if(self.reconstruction==2):
      print("Reconstruction: PLM")
    elif(self.reconstruction==3):
      print("Reconstruction: LimO3")
    elif(self.reconstruction==4):
      print("Reconstruction: PPM")
    if(self.vectPot):
      print("Vector Potential: ON")
    else:
      print("Vector Potential: OFF")
    if self.mpi:
      print("MPI: ON")
    else:
      print("MPI: OFF")

    print("**************************************************************")

  def _getReferenceFilename(self):
    strReconstruction="plm"
    if self.reconstruction == 3:
      strReconstruction = "limo3"
    if self.reconstruction == 4:
      strReconstruction= "ppm"

    strPrecision="double"
    if self.single:
      strPrecision="single"

    fileref='dump.ref.'+strPrecision+"."+strReconstruction+"."+self.inifile
    if self.vectPot:
      fileref=fileref+".vectPot"

    fileref=fileref+'.dmp'
    return(fileref)

  def _computeError(self,Vref,Vtest):
    ntested=0
    error=0
    for fld in Vtest.data.keys():
      if(Vtest.data[fld].ndim==3):
        if fld in Vref.data.keys():
          #print("error in "+fld+" = "+str(np.sqrt(np.mean((Vref.data[fld]-Vtest.data[fld])**2))))
          error = error+np.sqrt(np.mean((Vref.data[fld]-Vtest.data[fld])**2))
          ntested=ntested+1

    if ntested==0:
      raise Exception(bcolors.FAIL+"There is no common field between the reference and current file"+bcolors.ENDC)

    error=error/ntested
    return(error)

  def _plotDiff(self,Vref,Vtest):

    for fld in Vtest.data.keys():
      if(Vtest.data[fld].ndim==3):
        if fld in Vref.data.keys():
          plt.figure()
          plt.title(fld)
          x1=Vref.x1
          if Vref.data[fld].shape[0] == Vref.x1.size+1:
            x1=np.zeros(Vref.data[fld].shape[0])
            x1[:-1]=Vref.x1l
            x1[-1]=Vref.x1r[-1]
          x2=Vref.x2
          if Vref.data[fld].shape[1] == Vref.x2.size+1:
            x2=np.zeros(Vref.data[fld].shape[1])
            x2[:-1]=Vref.x2l
            x2[-1]=Vref.x2r[-1]
          x3=Vref.x3
          if Vref.data[fld].shape[2] == Vref.x3.size+1:
            x3=np.zeros(Vref.data[fld].shape[2])
            x3[:-1]=Vref.x3l
            x3[-1]=Vref.x3r[-1]
          if Vref.data[fld].shape[1]>1:
            plt.pcolor(x1, x2, Vref.data[fld][:,:,0].T-Vtest.data[fld][:,:,0].T,cmap='seismic')
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.colorbar()
          else:
            plt.plot(x1, Vref.data[fld][:,0,0]-Vtest.data[fld][:,0,0])
            plt.xlabel("x1")
    plt.show()
