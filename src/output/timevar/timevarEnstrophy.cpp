
// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <iostream>
#include <fstream>
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "field.hpp"
#include "input.hpp"
#include "global.hpp"
#include "reduce.hpp"
#include "grid.hpp"
#include "loop.hpp"
#include "fft.hpp"
#include "timevarEnstrophy.hpp"

void TimeVarEnstrophy::Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) {
    astra::pushRegion("TimeVarEnstrophy::Write");
    // Compute enstrophy
    real enstrophy = 0.0;
    Array3D<complex> vx1, vx2, vx3;
    if(this->name.compare("w2") == 0) {
      vx1 = field["vx1"];
      vx2 = field["vx2"];
      vx3 = field["vx3"];
    } else if(this->name.compare("j2") == 0) {
      vx1 = field["bx1"];
      vx2 = field["bx2"];
      vx3 = field["bx3"];
    }

    // Compute curl of input field
    Array3D<complex> wx1 = astra::makeArray<Array3D<complex>>("w_x1", grid->npf);
    Array3D<complex> wx2 = astra::makeArray<Array3D<complex>>("w_x2", grid->npf);
    Array3D<complex> wx3 = astra::makeArray<Array3D<complex>>("w_x3", grid->npf);
    auto kx1 = this->grid->kx[IDIR];
    auto kx2 = this->grid->kx[JDIR];
    auto kx3 = this->grid->kx[KDIR];

    astra_for("compute_curl_wx", field,
      KOKKOS_LAMBDA(int64_t i,int64_t j,int64_t k) {
        Kokkos::complex ikx = Kokkos::complex(0.0, kx1(i));
        Kokkos::complex iky = Kokkos::complex(0.0, kx2(j));
        Kokkos::complex ikz = Kokkos::complex(0.0, kx3(k));
        wx1(i,j,k) = iky * vx3(i,j,k) - ikz * vx2(i,j,k);
        wx2(i,j,k) = ikz * vx1(i,j,k) - ikx * vx3(i,j,k);
        wx3(i,j,k) = ikx * vx2(i,j,k) - iky * vx1(i,j,k);
    });

    Array3D<real> wr1 = astra::makeArray<Array3D<real>>("w1_real", grid->npr);
    Array3D<real> wr2 = astra::makeArray<Array3D<real>>("w2_real", grid->npr);
    Array3D<real> wr3 = astra::makeArray<Array3D<real>>("w3_real", grid->npr);
    // Transform to real space
    grid->fft->C2R(wx1, wr1);
    grid->fft->C2R(wx2, wr2);
    grid->fft->C2R(wx3, wr3);

    // Compute enstrophy
    astra_reduce("timevar_enstrophy_compute_", fieldReal,
        KOKKOS_LAMBDA(int64_t i,int64_t j,int64_t k, real& local_sum) {
          local_sum += 0.5 * ( wr1(i,j,k) * wr1(i,j,k)
                             + wr2(i,j,k) * wr2(i,j,k)
                             + wr3(i,j,k) * wr3(i,j,k) );
      }, Kokkos::Sum<real>(enstrophy));

    int64_t ntot = grid->npr_glob[IDIR];
    ntot *= grid->npr_glob[JDIR];
    ntot *= grid->npr_glob[KDIR];
    enstrophy /= ntot;
    #ifdef WITH_MPI
      // Reduce across all processes
      real q0;
      MPI_Reduce(&enstrophy, &q0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      enstrophy = q0;
    #endif
    // Write to file
    if(this->isRoot) {
      std::ofstream outfile;
      outfile.open(this->filename, std::ios_base::app);
      outfile.precision(10);
      outfile << std::scientific << t << "\t" << enstrophy << std::endl;
      outfile.close();
    }
    astra::popRegion();
  }
