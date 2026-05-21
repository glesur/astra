// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic interface for energy statistics

#ifndef OUTPUT_TIMEVAR_TIMEVARSPECTRUM_HPP_
#define OUTPUT_TIMEVAR_TIMEVARSPECTRUM_HPP_

#include <string>
#include <memory>
#include "timevar.hpp"
#include "arrays.hpp"
#include "linearshear.hpp"

class TimeVarSpectrum : public TimeVar {
 public:
  TimeVarSpectrum(Input &input, Grid *grid, std::string name, std::string directory);

  ~TimeVarSpectrum() override {}
  void Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) override;

 private:
  std::string var1, var2;
  int nbins{0};
  real kmin{0.0}, kmax{0.0};
  bool haveLinearShear{false};
  std::unique_ptr<LinearShear> linearShear;
};


#endif // OUTPUT_TIMEVAR_TIMEVARSPECTRUM_HPP_
