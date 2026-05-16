// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include "output.hpp"
#include "dump.hpp"
#include "dumpVariables.hpp"
#include "timevar.hpp"
#include "vtk.hpp"
#include "grid.hpp"
#include "input.hpp"
#include "field.hpp"
#include "loop.hpp"
#include "linearshear.hpp"

Output::~Output() = default;    // Destructor needed for unique_ptr of timeVarOutput

Output::Output(Input &input, Grid &grid) {
  this->grid = &grid;

  // Initialise vtk output variables
  outputVtkStep = input.GetOrSet<real>("Output","vtk",0,0.1);
  lastVtkOutput = -outputVtkStep; // so that we output at time 0.0
  
  if(input.CheckEntry("Output","vtk_dir")>=0) {
    outputVtkDirectory = input.Get<std::string>("Output","vtk_dir",0);
  }
  nvtk = 0;

  int nAdditionalVariables = input.CheckEntry("Output","vtk_addvar");
  if(nAdditionalVariables>0) {
    // We have additional variables to output in vtk files, read them
    vtkAdditionalVariables = Field<Array3D<complex>>("vtk_additional_variables", grid.npf);
    for(int i=0; i<nAdditionalVariables; i++) {
      std::string varName = input.Get<std::string>("Output","vtk_addvar",i);
      vtkAdditionalVariableNames.push_back(varName);
      astra::cout << "Output: Additional variable \"" << varName << "\" in vtk output." << std::endl;
      if(varName.compare("j")==0) {
        vtkAdditionalVariables.Add("jx1");
        vtkAdditionalVariables.Add("jx2");
        vtkAdditionalVariables.Add("jx3");
      } else if(varName.compare("w")==0) {
        vtkAdditionalVariables.Add("wx1");
        vtkAdditionalVariables.Add("wx2");
        vtkAdditionalVariables.Add("wx3");
      } else {
        std::stringstream msg;
        msg << "Unknown additional variable name \"" << varName << "\" for vtk output." << std::endl
            << "Available additional variables are: \"j\" for current density and \"w\" for vorticity.";
        throw std::runtime_error(msg.str());
      }
    }
  }

  // Initialise dump output variables
  outputDmpStep = input.GetOrSet<real>("Output","dump",0,-1);
  lastDmpOutput = -outputDmpStep; // so that we output at time 0.0
  outputDmpDir = input.GetOrSet<std::string>("Output","dump_dir",0,"./");
  ndmp = 0;

  // Whether we want timevar
  timeVarOutput = std::make_unique<TimeVarOutput>(input, &grid);
  lastTimevar = -timevarStep; // so that we output at time 0.0
  timevarStep = input.GetOrSet<real>("Output","timevar_step",0,-1.0);

  std::string shearTypeStr = input.GetOrSet<std::string>("Physics","shear_type",0,"disabled");
  if(shearTypeStr =="linear") {
    this->haveShear = true;
    this->linearShear = std::make_unique<LinearShear>(input, &grid);
  }

  // Declare what we need as dump variables for restarts

  DumpVariables::Register("lastVtkOutput", lastVtkOutput);
  DumpVariables::Register("lastDmpOutput", lastDmpOutput);
  DumpVariables::Register("lastTimevar", lastTimevar);
  DumpVariables::Register("nvtk", nvtk);
  DumpVariables::Register("ndmp", ndmp);
    
}

void Output::RestartFromDump(Input &input) {
  ///////////////////////////////
  // Restarting from dump file
  ///////////////////////////////
  int restartFileNumber = input.restartFileNumber;
  if(restartFileNumber==-1) {
    // Find the last dump file in the output directory
    restartFileNumber = GetLastDumpInDirectory(outputDmpDir);
    if(restartFileNumber<0) {
      throw std::runtime_error("No dump file found in directory "+outputDmpDir+" for restart.");
    }
  }
  std::string filename;

  if(restartFileNumber==-2) {
    // Restart from Snoopy dump
    filename = input.Get<std::string>("CommandLine","snoopy_restart",0);
    astra::cout << "Output: Restarting from Snoopy dump file " << filename << std::endl;
  } else {
    filename = outputDmpDir+"/"+CreateDumpFileName(restartFileNumber);
    astra::cout << "Output: Restarting from dump file number " << restartFileNumber << std::endl;
  }
  
  Dump dump(grid, filename);
  if(restartFileNumber==-2) {
    dump.ReadSnoopy();
  } else {
    dump.Read();
  }
  DumpVariables::FetchFromDump(&dump);
}


void Output::CheckForOutput(Input &input, Field<Array3D<complex>> state, real time) {
  if(outputVtkStep>=0.0 && time-lastVtkOutput>=outputVtkStep) {
    std::stringstream ssfileName, ssvtkFileNum;
    ssvtkFileNum << std::setfill('0') << std::setw(4) << nvtk;
    std::string filename = std::string("data.")+ssvtkFileNum.str();
    Vtk vtk(grid, input, time, filename,outputVtkDirectory);
    vtk.Write(state, time);
    if(!vtkAdditionalVariableNames.empty()) {
      ComputeAdditionalVariables(state, time);
      vtk.Write(vtkAdditionalVariables, time);
    }
    nvtk++;
    lastVtkOutput += outputVtkStep;
  }
  if(timevarStep>0.0 && time-lastTimevar>=timevarStep) {
    if(time==0.0) {
      timeVarOutput->Reset();
    }
    timeVarOutput->Write(state, time);
    lastTimevar += timevarStep;
  }
  if(outputDmpStep>=0.0 && time-lastDmpOutput>=outputDmpStep) {
    ForceDump(time);
  }
}

void Output::ForceDump(real time) {
  Dump dump(grid, outputDmpDir+"/"+CreateDumpFileName(ndmp));
  ndmp++;
  lastDmpOutput += outputDmpStep;
  DumpVariables::StoreToDump(&dump);
  dump.Write();
}

int Output::GetLastDumpInDirectory(std::string directory_str) {
  int num = -1;

  fs::path directory = directory_str;
  std::time_t youngFileTime;
  bool first = true;
  for (const auto & entry : fs::directory_iterator(directory)) {
      // Check file extension
      if(entry.path().extension().string().compare(".dmp")==0) {
        auto fileTime = to_time_t(fs::last_write_time(entry.path()));
        // Check which one is the most recent
        if(first || fileTime>youngFileTime) {
          // std::tm *gmt = std::gmtime(&fileTime);
          // idfx::cout << "file " << entry.path() << "is the most recent with "
          //            << std::put_time(gmt, "%d %B %Y %H:%M:%S") << std::endl;

          // Ours is more recent, extract the dump file number
          try {
            num = std::stoi(entry.path().filename().string().substr(5,4));
            first = false;
            youngFileTime = fileTime;
          } catch (...) {
            // the file name does not follow the convention "filebase.xxxx.dmp"
            // ->We skip it
          }
        }
      }
  }
  return(num);
}

std::string Output::CreateDumpFileName(int n) {
  std::stringstream ssdmpFileNum;
  ssdmpFileNum << std::setfill('0') << std::setw(4) << n;
  return std::string("dump.")+ssdmpFileNum.str()+std::string(".dmp");
}

void Output::ComputeAdditionalVariables(Field<Array3D<complex>> state, real time) {
  // Compute any additional variables that we want to output but are not part of the state vector
  for(auto & variable : vtkAdditionalVariableNames) {
    if(variable.compare("j")==0) {
      // Compute current density j = curl(b)
      ComputeCurl({vtkAdditionalVariables["jx1"], vtkAdditionalVariables["jx2"], vtkAdditionalVariables["jx3"]},
                  {state["bx1"], state["bx2"], state["bx3"]}, time);
    } else if(variable.compare("w")==0) {
      // Compute vorticity w = curl(v)
      ComputeCurl({vtkAdditionalVariables["wx1"], vtkAdditionalVariables["wx2"], vtkAdditionalVariables["wx3"]},
                  {state["vx"], state["vy"], state["vz"]}, time);
    } else {
      throw std::runtime_error("Output: Additional variable \""+variable+"\" not implemented.");
    }
  }
}

// Compute out = i (k1 in2 - k2 in1), where k1 and k2 are the wavevector components along the two directions perpendicular to the direction of the output variable out (e.g. for out=jx1, k1=kx2 and k2=kx3)
void Output::ComputeCurl(std::array<Array3D<complex>, 3> out , std::array<Array3D<complex>, 3> in, real time) {
  auto kx1 = grid->kx[IDIR];
  auto kx2 = grid->kx[JDIR];
  auto kx3 = grid->kx[KDIR];

  auto in1 = in[0];
  auto in2 = in[1];
  auto in3 = in[2];
  auto out1 = out[0];
  auto out2 = out[1];
  auto out3 = out[2];
  
  bool haveShear = this->haveShear;
  LinearShear shear = *(this->linearShear.get());
  if(haveShear) {
    shear.SetTinit(time);
  }
  astra_for("output_compute_curl", 0,grid->npf[IDIR],0,grid->npf[JDIR],0,grid->npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      real k1, k2, k3;
      if(haveShear) {
        k1 = shear.kx1t(kx1(i),kx2(j),kx3(k));
        k2 = shear.kx2t(kx1(i),kx2(j),kx3(k));
        k3 = shear.kx3t(kx1(i),kx2(j),kx3(k));
      } else {
        k1 = kx1(i);
        k2 = kx2(j);
        k3 = kx3(k);
      }
      out1(i,j,k) = Kokkos::complex(0.0,1.0) * (k2*in3(i,j,k) - k3*in2(i,j,k));
      out2(i,j,k) = Kokkos::complex(0.0,1.0) * (k3*in1(i,j,k) - k1*in3(i,j,k));
      out3(i,j,k) = Kokkos::complex(0.0,1.0) * (k1*in2(i,j,k) - k2*in1(i,j,k));
  });
}




