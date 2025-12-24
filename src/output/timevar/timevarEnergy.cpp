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
#include "timevarEnergy.hpp"
#include <iostream>
#include <fstream>
#ifdef WITH_MPI
#include <mpi.h>
#endif

void TimeVarEnergy::Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) {
    astra::pushRegion("TimeVarEnergy::Write");
    // Compute energy
    real energy_total = 0.0;
    std::vector<std::string> vars;
    if(this->name.compare("ev") == 0) {
      vars = {"vx1","vx2","vx3"};
    } else if(this->name.compare("eb") == 0) {
      vars = {"bx1","bx2","bx3"};
    }

    for(auto& var : vars) {
      real energy = 0.0;
      auto view = fieldReal[var];
      astra_reduce("timevar_energy_compute_"+var, fieldReal,
        KOKKOS_LAMBDA(int i,int j,int k, real& local_sum) {
          local_sum += 0.5 * view(i,j,k) * view(i,j,k);
      }, Kokkos::Sum<real>(energy));
      energy_total += energy;
    }
    energy_total *= grid->dx[IDIR]*grid->dx[JDIR]*grid->dx[KDIR];
    #ifdef WITH_MPI
      // Reduce across all processes
      real q0;
      MPI_Reduce(&energy_total, &q0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      energy_total = q0;
    #endif
    // Write to file
    if(this->isRoot) {
      std::ofstream outfile;
      outfile.open(this->filename, std::ios_base::app);
      outfile.precision(10);
      outfile << std::scientific << t << "\t" << energy_total << std::endl;
      outfile.close();
    }
    astra::popRegion();
  }