// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <string>

#include "input.hpp"
#include "grid.hpp"
#include "global.hpp"
#include "arrays.hpp"



Grid::Grid(Input &input) {
  astra::pushRegion("Grid::Grid(Input)");


  // Get grid size from input file, block [Grid]
  int npoints[3];
  for(int dir = 0 ; dir < 3 ; dir++) {
    std::string label = std::string("X")+std::to_string(dir+1)+std::string("-grid");

    np[dir] = input.Get<int>("Grid",label,1);
    xbeg[dir] = input.Get<real>("Grid",label,0);
    xend[dir] = input.Get<real>("Grid",label,2);

    x[dir] = Array1D<real>("Grid_x",np[dir]);
    xr[dir] = Array1D<real>("Grid_xr",np[dir]);
    xl[dir] = Array1D<real>("Grid_xl",np[dir]);

    dx[dir] = (xend[dir] - xbeg[dir])/np[dir];
  }

  // Initialise the grid elements
  for(int dir = 0 ; dir < 3 ; dir++) {
    Array1D<real> x = this->x[dir];
    Array1D<real> xl = this->xl[dir];
    Array1D<real> xr = this->xr[dir];
    real dx = this->dx[dir];
    real xb = this->xbeg[dir];
    real xe = this->xend[dir];
    astra_for("Init grid array", 0, np[dir],
      KOKKOS_LAMBDA(int i) {
        xl(i) = xb + i*dx;
        xr(i) = xb + (i+1)*dx;
        x(i) = xb + (i+1./2.)*dx;
      });
  }

   astra::popRegion();
}


void Grid::ShowConfig() {
  astra::cout << "Grid: full grid size is " << std::endl;
  for(int dir = 0 ; dir < 3 ; dir++) {
    

      astra::cout << "\t Direction X" << (dir+1) << ": " << "\t" << xbeg[dir]
                 << "...." << np[dir] << "...." << xend[dir] << "\t"
                 << std::endl;
  }
  
}

