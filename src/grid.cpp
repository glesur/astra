// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <string>
#include <KokkosFFT.hpp>

#include "astra.hpp"
#include "loop.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "global.hpp"
#include "arrays.hpp"
#include "fft.hpp"




Grid::Grid(Input &input) {
  astra::pushRegion("Grid::Grid(Input)");
  // Get grid size from input file, block [Grid]
  for(int dir = 0 ; dir < 3 ; dir++) {
    std::string label = std::string("X")+std::to_string(dir+1)+std::string("-grid");

    npr_glob[dir] = input.Get<int>("Grid",label,1);
    xbeg_glob[dir] = input.Get<real>("Grid",label,0);
    xend_glob[dir] = input.Get<real>("Grid",label,2);

    npf_glob[dir] = npr_glob[dir];
    if(dir == KDIR) npf_glob[dir] = npf_glob[dir]/2+1;


    // todo: MPI domain decomposition
    // #ifundef MPI
    npr[dir] = npr_glob[dir];
    npf[dir] = npf_glob[dir];
    npr_t[dir] = npr[dir]; // by default no transposition
    #ifdef WITH_MPI
      if(dir == IDIR) {
        // We assume a decomposition along the first dimension only for now
        npr[dir] = npr_glob[dir]/astra::psize;
        npf[dir] = npf_glob[dir]/astra::psize;
      }
    #endif

    xbeg[dir] = xbeg_glob[dir] + astra::prank* (xend_glob[dir]-xbeg_glob[dir])/astra::psize;
    xend[dir] = xend_glob[dir];

    // #endif
    x_glob[dir] = Array1D<real>("Grid_x_glob",npr_glob[dir]);
    x[dir] = Array1D<real>("Grid_x",npr[dir]);

    dx[dir] = (xend_glob[dir] - xbeg_glob[dir])/npr_glob[dir];
    kmax[dir] = M_PI/dx[dir];
  }

  #ifdef WITH_MPI
    // Compute transposed sizes
    npr_t[IDIR] = npr_glob[JDIR]/astra::psize;
    npr_t[JDIR] = npr_glob[IDIR];
  #endif

  this->InitGrid();
   astra::popRegion();
}

// To allow for capture in Lambda functions, astra_for can't be used in class constructors
void Grid::InitGrid() {
  astra::pushRegion("Grid::InitGrid()");
  // Initialise the grid elements
  for(int dir = 0 ; dir < 3 ; dir++) {
    Array1D<real> x = this->x_glob[dir];

    real dx = this->dx[dir];
    real xb = this->xbeg_glob[dir];
    real xe = this->xend_glob[dir];
    astra_for("Init grid array", 0, npr_glob[dir],
      KOKKOS_LAMBDA(int i) {
        x(i) = xb + (i+1./2.)*dx;
      });

    x = this->x[dir];
    xb = this->xbeg[dir];
    xe = this->xend[dir];
    astra_for("Init grid array", 0, npr[dir],
      KOKKOS_LAMBDA(int i) {
        x(i) = xb + (i+1./2.)*dx;
      });
  }

  // initialise wavenumbers
  for(int dir = 0 ; dir < 3 ; dir++) {
    real d = (xend_glob[dir]-xbeg_glob[dir])/(2.0*M_PI*static_cast<real>(npr_glob[dir]));
    if(dir < KDIR) kx_glob[dir] = KokkosFFT::fftfreq(Device(), npr_glob[dir], d);
    else kx_glob[dir] = KokkosFFT::rfftfreq(Device(), npr_glob[dir], d);
  }

  // Get local wavenumbers
  for(int dir = 0 ; dir < 3 ; dir++) {
    kx[dir] = Array1D<real>("Grid_kx",npf[dir]);
    auto k = kx[dir];
    auto kglob = kx_glob[dir];
    const int offset = (dir==IDIR) ? astra::prank* (npf_glob[dir])/astra::psize : 0;

    astra_for("Init k array", 0, npf[dir],
    KOKKOS_LAMBDA(int i) {
      k(i) = kglob(i+offset);
    });
  }
  // Nothing to do here for now
  this->fft = std::make_unique<FFT>(npr_glob, npf_glob);

  astra::popRegion();
}

void Grid::ShowConfig() {
  astra::cout << "Grid: full grid size is " << std::endl;
  for(int dir = 0 ; dir < 3 ; dir++) {
      astra::cout << "\t Direction X" << (dir+1) << ": " << "\t" << xbeg_glob[dir]
                 << "...." << npr_glob[dir] << "...." << xend_glob[dir] << "\t"
                 << std::endl;
  }
  
}

