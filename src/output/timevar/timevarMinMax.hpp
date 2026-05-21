// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic interface for energy statistics

#ifndef OUTPUT_TIMEVAR_TIMEVARMINMAX_HPP_
#define OUTPUT_TIMEVAR_TIMEVARMINMAX_HPP_

#include <string>
#include "timevar.hpp"
#include "arrays.hpp"

class TimeVarMinMax : public TimeVar {
 public:
  TimeVarMinMax(Input &input, Grid *grid, std::string name, std::string directory);
  ~TimeVarMinMax() override {}
  void Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) override;
 private:
  bool isMax{true}; // otherwise it's min
  std::string varname;
};


#endif // OUTPUT_TIMEVAR_TIMEVARMINMAX_HPP_
