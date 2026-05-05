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


void TimeVarStress::Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) {
    astra::pushRegion("TimeVarStress::Write");
    // Compute stress
    real stress = 0.0;

    auto view1 = fieldReal[var1];
    auto view2 = fieldReal[var2];
    astra_reduce("timevar_stress_compute_"+name, fieldReal,
        KOKKOS_LAMBDA(int64_t i,int64_t j,int64_t k, real& local_sum) {
          local_sum += view1(i,j,k) * view2(i,j,k);
      }, Kokkos::Sum<real>(stress));

    stress /= grid->npr_glob[IDIR]*grid->npr_glob[JDIR]*grid->npr_glob[KDIR];
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