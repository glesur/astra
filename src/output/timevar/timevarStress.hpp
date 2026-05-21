// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic interface for energy statistics

#ifndef OUTPUT_TIMEVAR_TIMEVARSTRESS_HPP_
#define OUTPUT_TIMEVAR_TIMEVARSTRESS_HPP_

#include <string>
#include "timevar.hpp"
#include "arrays.hpp"

class TimeVarStress : public TimeVar {
 public:
  TimeVarStress(Input &input, Grid *grid, std::string name, std::string directory) : TimeVar(input, grid, name, directory) {
    size_t dotLocation = this->name.find(".");
    var1 = this->name.substr(0, dotLocation);
    var2 = this->name.substr(dotLocation+1);

    // translate variable names from snoopy
    var1 = this->ExtractVarName(var1);
    var2 = this->ExtractVarName(var2);
  }

  ~TimeVarStress() override {}
  void Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) override;
 private:
  std::string var1, var2;
};


#endif // OUTPUT_TIMEVAR_TIMEVARSTRESS_HPP_
