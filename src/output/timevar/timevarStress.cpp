// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************


#include "field.hpp"
#include "input.hpp"
#include "global.hpp"
#include "reduce.hpp"
#include "grid.hpp"
#include "timevarStress.hpp"
#include <iostream>
#include <fstream>
#ifdef WITH_MPI
#include <mpi.h>
#endif

std::string extractVarName(const std::string& name) {
  std::string varname;
  if(name == "vx") varname = "vx1";
  else if(name == "vy") varname = "vx2";
  else if(name == "vz") varname = "vx3";
  else if(name == "bx") varname = "bx1";
  else if(name == "by") varname = "bx2";
  else if(name == "bz") varname = "bx3";
  else varname = name;
  return varname;
}
void TimeVarStress::Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) {
    astra::pushRegion("TimeVarStress::Write");
    // Compute energy
    real energy_total = 0.0;
    std::string var1, var2;

    size_t dotLocation = this->name.find(".");
    var1 = this->name.substr(0, dotLocation);
    var2 = this->name.substr(dotLocation+1);
    
    // translate variable names from snoopy
    var1 = extractVarName(var1);
    var2 = extractVarName(var2);

    real stress = 0.0;

    auto view1 = fieldReal[var1];
    auto view2 = fieldReal[var2];
    astra_reduce("timevar_stress_compute_"+name, fieldReal,
        KOKKOS_LAMBDA(int i,int j,int k, real& local_sum) {
          local_sum += view1(i,j,k) * view2(i,j,k);
      }, Kokkos::Sum<real>(stress));

    stress *= grid->dx[IDIR]*grid->dx[JDIR]*grid->dx[KDIR];
    #ifdef WITH_MPI
      // Reduce across all processes
      real q0;
      MPI_Reduce(&stress, &q0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      stress = q0;
    #endif
    // Write to file
    if(this->isRoot) {
      std::ofstream outfile;
      outfile.open(this->filename, std::ios_base::app);
      outfile.precision(10);
      outfile << std::scientific << t << "\t" << stress << std::endl;
      outfile.close();
    }
    astra::popRegion();
  }