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

Output::~Output() = default;    // Destructor needed for unique_ptr of timeVarOutput

Output::Output(Input &input, Grid &grid) {
  this->grid = &grid;

  // Initialise output variables
  lastVtkOutput = 0.0;
  outputVtkStep = input.GetOrSet<real>("Output","vtk",0,0.1);
  
  if(input.CheckEntry("Output","vtk_dir")>=0) {
    outputVtkDirectory = input.Get<std::string>("Output","vtk_dir",0);
  }
  nvtk = 0;

  lastDmpOutput = 0.0;
  outputDmpStep = input.GetOrSet<real>("Output","dump",0,-1);
  outputDmpDir = input.GetOrSet<std::string>("Output","dump_dir",0,"./");
  ndmp = 0;

  // Whether we want timevar
  timeVarOutput = std::make_unique<TimeVarOutput>(input, &grid);
  lastTimevar = 0.0;
  timevarStep = input.GetOrSet<real>("Output","timevar_step",0,-1.0);

  // Declare what we need as dump variables for restarts

  DumpVariables::Register("lastVtkOutput", lastVtkOutput);
  DumpVariables::Register("lastDmpOutput", lastDmpOutput);
  DumpVariables::Register("lastTimevar", lastTimevar);
  DumpVariables::Register("nvtk", nvtk);
  DumpVariables::Register("ndmp", ndmp);

  if(input.restartRequested) {
    ///////////////////////////////
    // Restarting from dump file
    ///////////////////////////////
    int restartFileNumber = input.restartFileNumber;;
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
    
    Dump dump(&grid, filename);
    if(restartFileNumber==-2) {
      dump.ReadSnoopy();
    } else {
      dump.Read();
    }
    DumpVariables::FetchFromDump(&dump);
  }
}

void Output::CheckForOutput(Input &input, Field<Array3D<complex>> state, real time) {
  if(outputVtkStep>0.0 && time-lastVtkOutput>=outputVtkStep) {
    std::stringstream ssfileName, ssvtkFileNum;
    ssvtkFileNum << std::setfill('0') << std::setw(4) << nvtk;
    std::string filename = std::string("data.")+ssvtkFileNum.str();
    Vtk vtk(grid, input, time, filename,outputVtkDirectory);
    vtk.Write(state, time);
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
  if(outputDmpStep>0.0 && time-lastDmpOutput>=outputDmpStep) {
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






