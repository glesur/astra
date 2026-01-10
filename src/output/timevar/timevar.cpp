// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include "timevar.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "input.hpp"
#include "global.hpp"
#include "timevarEnergy.hpp"
#include "timevarEnstrophy.hpp"
#include "timevarMinMax.hpp"
#include "timevarStress.hpp"
#include "fft.hpp"

#if __has_include(<filesystem>)
  #include <filesystem> // NOLINT [build/c++17]
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include <limits>
#include <vector>

TimeVar::TimeVar(Input &input, Grid *grid, std::string name, std::string directory) : grid(grid), name(name) {
  this->isRoot = astra::prank == 0;
  fs::path outputDirectory = directory;
  std::stringstream ssfileName;
  ssfileName << name << ".dat";
  this->filename = outputDirectory / ssfileName.str();
}
  
void TimeVar::Reset() {
  // Remove files if they exist
  if(isRoot) {
    if(fs::exists(filename)) {
      fs::remove(filename);
    } 
  }
}
  
void TimeVarOutput::Write(Field<Array3D<complex>>& field, const real t) {
  Field<Array3D<real>> fieldReal("tempReal", grid->npr);
  for(auto const& [name, view] : field) {
    fieldReal.Add(name);
    auto viewReal = fieldReal[name];
    this->grid->fft->C2R(view, viewReal);
  }
  for(auto& timevar : this->timevarList) {
    timevar->Write(t, field, fieldReal);
  }
}

void TimeVarOutput::Reset() {
  for(auto& timevar : this->timevarList) {
    timevar->Reset();
  }
}

TimeVarOutput::TimeVarOutput(Input &input, Grid *grid) {
  this->grid = grid;
  const int nvars = input.CheckEntry("Output","timevar");
  if(nvars <= 0) {
    astra::cout << "TimeVarOutput: No timevar variables requested." << std::endl;
    return;
  }
  std::string directory = input.GetOrSet<std::string>("Output","timevar_dir",0,"timevar");
  
  // Create output directory if it does not exist
  fs::path outputDirectory = directory;
  if(astra::prank == 0) {
    if(!fs::is_directory(outputDirectory)) {
      try {
        if(!fs::create_directory(outputDirectory)) {
          std::stringstream msg;
          msg << "Cannot create directory " << outputDirectory << std::endl;
          throw std::runtime_error(msg.str());
        }
      } catch(std::exception &e) {
        std::stringstream msg;
        msg << "Cannot create directory " << outputDirectory << std::endl;
        msg << e.what();
        throw std::runtime_error(msg.str());
      }
    }
  }

  // Loop over variables
  for(int i=0; i<nvars; i++) {
    std::string varname = input.Get<std::string>("Output","timevar",i);
    // Create timevar object
    switch(str2int(varname)) {
      case str2int("ev"):
        timevarList.push_back(std::make_unique<TimeVarEnergy>(input, grid, varname, directory));
        break;
      case str2int("vxmax"):
      case str2int("vxmin"):
      case str2int("vymax"):
      case str2int("vymin"):
      case str2int("vzmax"):
      case str2int("vzmin"):
        timevarList.push_back(std::make_unique<TimeVarMinMax>(input, grid, varname, directory));
        break;
      case str2int("w2"):
      case str2int("j2"):
        timevarList.push_back(std::make_unique<TimeVarEnstrophy>(input, grid, varname, directory));
        break;
      default:
        // Check it contains a dot, in which case it's a stress tensor component
        if(varname.find(".") != std::string::npos) {
          timevarList.push_back(std::make_unique<TimeVarStress>(input, grid, varname, directory));
          break;
        }
        std::stringstream msg;
        msg << "Unknown timevar variable requested: " << varname << std::endl;
        throw std::runtime_error(msg.str());
    }
  }

}
