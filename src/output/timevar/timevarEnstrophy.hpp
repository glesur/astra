// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic interface for enstrophy/current statistics

#ifndef OUTPUT_TIMEVAR_TIMEVARENSTROPHY_HPP_
#define OUTPUT_TIMEVAR_TIMEVARENSTROPHY_HPP_

#include "timevar.hpp"
#include "arrays.hpp"

class TimeVarEnstrophy : public TimeVar {
 public:
  TimeVarEnstrophy(Input &input, Grid *grid, std::string name, std::string directory) : TimeVar(input, grid, name, directory) {};
  ~TimeVarEnstrophy() override {}
  void Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) override;
};


#endif // OUTPUT_TIMEVAR_TIMEVARENSTROPHY_HPP_