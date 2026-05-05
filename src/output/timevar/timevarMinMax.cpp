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
#include "timevarMinMax.hpp"
#include <iostream>
#include <fstream>
#ifdef WITH_MPI
#include <mpi.h>
#endif

TimeVarMinMax::TimeVarMinMax(Input &input, Grid *grid, std::string name, std::string directory) : TimeVar(input, grid, name, directory) {
  if(name.substr(2,3).compare("max") == 0) {
    isMax = true;
  } else if(name.substr(2,3).compare("min") == 0) {
    isMax = false;
  } else {
    throw std::runtime_error("Unknown TimeVarMinMax type: " + name);
  }
  varname = this->ExtractVarName(name.substr(0,2));
}

void TimeVarMinMax::Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) {
    astra::pushRegion("TimeVarMinMax::Write");
    // Compute max/min of field
    real v = isMax ? -std::numeric_limits<real>::infinity() : std::numeric_limits<real>::infinity();

    auto view = fieldReal[varname];
    if(isMax) {
      astra_reduce("timevar_max_compute_"+varname, fieldReal,
        KOKKOS_LAMBDA(int64_t i,int64_t j,int64_t k, real& myMax) {
          myMax = fmax(myMax, view(i,j,k));
      }, Kokkos::Max<real>(v));
      #ifdef WITH_MPI
        // Reduce across all processes
        real q0;
        MPI_Reduce(&v, &q0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        v = q0;
      #endif
    } else {
      astra_reduce("timevar_min_compute_"+varname, fieldReal,
        KOKKOS_LAMBDA(int64_t i,int64_t j,int64_t k, real& myMin) {
          myMin = fmin(myMin, view(i,j,k));
      }, Kokkos::Min<real>(v));
      #ifdef WITH_MPI
        // Reduce across all processes
        real q0;
        MPI_Reduce(&v, &q0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        v = q0;
      #endif
    }

    // Write to file
    if(this->isRoot) {
      std::ofstream outfile;
      outfile.open(this->filename, std::ios_base::app);
      outfile.precision(10);
      outfile << std::scientific << t << "\t" << v << std::endl;
      outfile.close();
    }
    astra::popRegion();
  }