// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <dirent.h>

#include <fstream>
#include <string>
#include <csignal>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "input.hpp"
#include "version.hpp"
#include "profiler.hpp"
#include "global.hpp"

// Flag will be set if a signal has been received
bool Input::abortRequested = false;

Input::Input() {
}

// Create input from file filename
// This routine expects input file of the following form:
// [Blockname]                                # comments2
// Parameter_name   Parametervalue1   Parametervalue2  Parametervalue3... # comments1
//
// Comments are allowed everywhere. Anything after # is ignored in the line
// Blockname should refer to one of Idefix class which will use the parameters
// in said block. Everything is stored in a map of maps of vectors of strings :-)

Input::Input(int argc, char* argv[] ) {
  std::ifstream file;
  std::string line, lineWithComments, blockName, paramName, paramValue;
  std::size_t firstChar, lastChar;
  bool haveBlock = false;
  std::stringstream msg;
  int nParameters = 0;    // # of parameters in current block

  // Tell the system we want to catch the SIGUSR2 signals
  signal(SIGUSR2, signalHandler);

  // Tell the time when input was initialised
  timer.reset();
  lastStopFileCheck = timer.seconds();

  // Default input file name
  this->inputFileName = std::string("astra.ini");

  // Parse command line (may replace the input file)
  ParseCommandLine(argc,argv);

  file.open(this->inputFileName);

  if(!file) {
    msg << "Input: cannot open input file " << this->inputFileName;
    throw std::runtime_error(msg.str());
  }

  while(std::getline(file, lineWithComments)) {
    line = lineWithComments.substr(0, lineWithComments.find("#",0));
    if (line.empty()) continue;     // skip blank line
    firstChar = line.find_first_not_of(" ");
    if (firstChar == std::string::npos) continue;      // line is all white space

    if (line.compare(firstChar, 1, "[") == 0) {        // a new block
      firstChar++;
      lastChar = (line.find_first_of("]", firstChar));

      if (lastChar == std::string::npos) {
        msg << "Block name '" << blockName << "' in file '"
            << this->inputFileName << "' not properly ended";
        throw std::invalid_argument(msg.str());
      }
      // Check if previous block was empty
      if(haveBlock && nParameters == 0) {
        throw std::invalid_argument(blockName+std::string(" block is empty in "+inputFileName));
        inputParameters[blockName]["!!empty!!"].push_back("!!empty!!");
      }
      blockName.assign(line, firstChar, lastChar-1);
      haveBlock = true;
      nParameters = 0;

      continue;   // Go to next line
    }   // New block

    // At this point, we should have a parameter set in the line
    if(haveBlock == false) {
      msg << "Input file '" << this->inputFileName
          << "' must specify a block name before the first parameter";
      throw std::invalid_argument(msg.str());
    }

    std::stringstream streamline(line);
    // Store the name of the parameter
    streamline >> paramName;
    nParameters++;
    // Store the parameters in parameter block
    while(streamline >> paramValue) {
      inputParameters[blockName][paramName].push_back(paramValue);
    }
  }
  file.close();
}

// This routine parse command line options
void Input::ParseCommandLine(int argc, char **argv) {
  std::stringstream msg;
  bool enableLogs = true;
  for(int i = 1 ; i < argc ; i++) {
    if(std::string(argv[i]) == "-restart") {
      std::string sirestart{};
      bool explicitDump = true;     // by default, assume a restart dump # was given
      // Check whether -restart was given with a number or not

      // -restart was the very last parameter
      if((i+1)>= argc) {
        explicitDump = false;
      } else if(std::isdigit(argv[i+1][0]) == 0) {
        // next argiment is another parameter (does not start with a number)
        explicitDump = false;
      }

      if(explicitDump) {
        sirestart = std::string(argv[++i]);
      } else {
        // implicitly restart from the latest existing dumpfile
        sirestart = "-1";
      }
      inputParameters["CommandLine"]["restart"].push_back(sirestart);
      this->restartRequested = true;
      this->restartFileNumber = std::stoi(sirestart);
    } else if(std::string(argv[i]) == "-snoopy_restart") {
      // -snoopy_restart must be followed by a filename
      if((i+1) >= argc) {
        throw std::invalid_argument(
                      "You must specify -snoopy_restart filename where filename is the name of the Snoopy dump file.");
      }
      std::string snoopyFileName = std::string(argv[++i]);
      inputParameters["CommandLine"]["snoopy_restart"].push_back(snoopyFileName);
      this->restartRequested = true;
      this->restartFileNumber = -2; // indicate a snoopy restart
    } else if(std::string(argv[i]) == "-i") {
      // Loop on dimensions
      if((++i) >= argc) throw std::invalid_argument(
                      "You must specify -i filename where filename is the name of the input file.");
      this->inputFileName = std::string(argv[i]);
    } else if(std::string(argv[i]) == "-maxcycles") {
      if((i+1)>= argc) {
        throw std::invalid_argument("-maxcycles requires an additional integer parameter");
      } else if(std::isdigit(argv[i+1][0]) == 0) {
        // next argiment is another parameter (does not start with a number)
        throw std::invalid_argument("-maxcycles requires an additional integer paramater");
      }
      this->maxCycles = std::stoi(std::string(argv[++i]));
      inputParameters["CommandLine"]["maxCycles"].push_back(std::to_string(maxCycles));
    } else if(std::string(argv[i]) == "-force_init") {
      this->forceInitRequested = true;
    } else if(std::string(argv[i]) == "-nowrite") {
      this->forceNoWrite = true;
      enableLogs = false;
    } else if(std::string(argv[i]) == "-nolog") {
      enableLogs = false;
    } else if(std::string(argv[i]) == "-profile") {
      astra::prof.EnablePerformanceProfiling();
    } else if(std::string(argv[i]) == "-Werror") {
      astra::warningsAreErrors = true;
    } else if(std::string(argv[i]) == "-version" || std::string(argv[i]) == "-v") {
      PrintVersion();
      astra::safeExit(0);
    } else if(std::string(argv[i]) == "-help" || std::string(argv[i]) == "-h") {
      PrintOptions();
      astra::safeExit(0);
    } else {
      PrintOptions();
      msg << "Unknown option " << argv[i];
      throw std::invalid_argument(msg.str());
    }
  }
  if(enableLogs) {
    astra::cout.enableLogFile();
  }
}


// This routine prints the parameters stored in the inputParameters structure
void Input::ShowConfig() {
  std::string blockName, paramName, paramValue;
  astra::cout << "-----------------------------------------------------------------------------"
             << std::endl;
  astra::cout << "Input Parameters using input file " << this->inputFileName << ":" << std::endl;
  astra::cout << "-----------------------------------------------------------------------------"
             << std::endl;
  for(InputContainer::iterator block = inputParameters.begin();
      block != inputParameters.end();
      block++ ) {
    blockName=block->first;
    astra::cout << "[" << blockName << "]" << std::endl;
    for(BlockContainer::iterator param = block->second.begin();
          param !=block->second.end(); param++) {
      paramName=param->first;
      astra::cout << "\t" << paramName << "\t";
      for(ParamContainer::iterator value = param->second.begin();
          value != param->second.end(); value++) {
        paramValue = *value;
        astra::cout << "\t" << paramValue;
      }
      astra::cout << std::endl;
    }
  }
  astra::cout << "-----------------------------------------------------------------------------"
             << std::endl;
  astra::cout << "-----------------------------------------------------------------------------"
             << std::endl;

  std::stringstream os;
  Kokkos::DefaultExecutionSpace().print_configuration(os, true);
  astra::cout << "Input: Kokkos configuration" << std::endl << os.str();
  astra::cout << "-----------------------------------------------------------------------------"
             << std::endl;

  #ifdef SINGLE_PRECISION
    astra::cout << "Input: Compiled with SINGLE PRECISION arithmetic." << std::endl;
  #else
    astra::cout << "Input: Compiled with DOUBLE PRECISION arithmetic." << std::endl;
  #endif
  #ifdef WITH_MPI
    astra::cout << "Input: MPI ENABLED." << std::endl;
  #endif
}

// This routine is called whenever a specific OS signal is caught
void Input::signalHandler(int signum) {
  astra::cout << std::endl << "Input: Caught interrupt " << signum << std::endl;
  abortRequested=true;
}

void Input::CheckForStopFile() {
  // Check whether a file "stop" has been created in directory. If so, raise the abort flag
  // Look for stop file every 5 seconds to avoid overloading the file system
  if(astra::prank==0) {
    std::string filename = std::string("stop");
    if(timer.seconds()-lastStopFileCheck>5.0) {
      lastStopFileCheck = timer.seconds();
      std::ifstream f(filename);
      if(f.good()) {
        // File exists, delete it and raise the flag
        std::remove(filename.c_str());
        abortRequested = true;
        astra::cout << std::endl << "Input: Caught stop file command" << std::endl;
      }
    }
  }
}

bool Input::CheckForAbort() {
  astra::pushRegion("Input::CheckForAbort");
  // Check whether an abort has been requesested
  // When MPI is present, we abort whenever one process got the signal
  CheckForStopFile();
#ifdef WITH_MPI
  int abortValue{0};
  bool returnValue{false};
  if(abortRequested) abortValue = 1;

  MPI_Bcast(&abortValue, 1, MPI_INT, 0, MPI_COMM_WORLD);
  returnValue = abortValue > 0;
  if(returnValue) astra::cout << "Input: CheckForAbort: abort has been requested." << std::endl;
  astra::popRegion();
  return(returnValue);
#else
  if(abortRequested) astra::cout << "Input: CheckForAbort: abort has been requested." << std::endl;
  astra::popRegion();
  return(abortRequested);
#endif
}

// Check that an entry is present in the ini file.
// If yes, return the number of parameters for given entry
int Input::CheckEntry(std::string blockName, std::string paramName) {
  int result=-1;
  InputContainer::iterator block = inputParameters.find(blockName);
  if(block != inputParameters.end()) {
    // Block exists
    BlockContainer::iterator param = block->second.find(paramName);
    if(param != block->second.end()) {
      // Parameter exist
      result = param->second.size();
    }
  }
  return(result);
}

// Check that a block is present in the ini file.
// If yes, return true
bool Input::CheckBlock(std::string blockName) {
  bool result = false;
  InputContainer::iterator block = inputParameters.find(blockName);
  if(block != inputParameters.end()) {
    // Block exists
    result = true;
  }
  return(result);
}

void Input::PrintLogo() {
  astra::cout << std::endl;
  PrintVersion();
  astra::cout << std::endl;
  astra::cout << std::endl;
}

void Input::PrintOptions() {
  astra::cout << "List of valid arguments:" << std::endl << std::endl;

  astra::cout << " -restart n" << std::endl;
  astra::cout << "         Restart from dumpfile n. If n is ommited, Astra restart from the latest"
             << " generated dump file." << std::endl;
  astra::cout << " -snoopy_restart filename" << std::endl;
  astra::cout << "         Restart from Snoopy dump \"filename\"." << std::endl;
  astra::cout << " -i xxx" << std::endl;
  astra::cout << "         Use the input file xxx instead of the default idefix.ini" << std::endl;
  astra::cout << " -maxcycles n" << std::endl;
  astra::cout << "         Perform at most n integration cycles." << std::endl;
  astra::cout << " -force_init" << std::endl;
  astra::cout << "         Call initial conditions before reading dump file ";
  astra::cout << "(this has no effect if -restart is not also passed)" << std::endl;
  astra::cout << " -nowrite" << std::endl;
  astra::cout << "         Do not generate any output file." << std::endl;
  astra::cout << " -nolog" << std::endl;
  astra::cout << "         Do not write any log file." << std::endl;
  astra::cout << " -profile" << std::endl;
  astra::cout << "         Enable on-the-fly performance profiling." << std::endl;
  astra::cout << " -Werror" << std::endl;
  astra::cout << "         Consider warnings as errors." << std::endl;
  astra::cout << " -v/-version" << std::endl;
  astra::cout << "         Show Idefix and kokkos version." << std::endl;
  astra::cout << " -h/-help" << std::endl;
  astra::cout << "         Show this message." << std::endl;
}

void Input::PrintVersion() {
  astra::cout << "              Astra version " << ASTRA_VERSION << std::endl;
  astra::cout << "              Built against Kokkos " << KOKKOS_VERSION << std::endl;
  astra::cout << "              Compiled on " << __DATE__ <<  " at " << __TIME__ << std::endl;
}
