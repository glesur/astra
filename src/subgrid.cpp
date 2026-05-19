// ***********************************************************************************
// Astra MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "subgrid.hpp"
#include "grid.hpp"
#include "input.hpp"
#include "arrays.hpp"
#include "global.hpp"


SubGrid::SubGrid() = default;
SubGrid::~SubGrid() = default;

SubGrid::SubGrid(Grid * grid, SliceType type, int dir, real x0):
          parentGrid(grid), sliceType(type), direction(dir) {

  // Copy everything from the parent grid, and then modify the relevant arrays
  this->x_glob = parentGrid->x_glob;
  this->x = parentGrid->x;
  this->dx = parentGrid->dx;
  this->kx_glob = parentGrid->kx_glob;
  this->kx = parentGrid->kx;
  this->kmax = parentGrid->kmax;
  this->xbeg_glob = parentGrid->xbeg_glob;
  this->xend_glob = parentGrid->xend_glob;
  this->xbeg = parentGrid->xbeg;
  this->xend = parentGrid->xend;
  this->npr_glob = parentGrid->npr_glob;
  this->npr = parentGrid->npr;
  this->npr_t = parentGrid->npr_t;
  this->npf_glob = parentGrid->npf_glob;
  this->npf = parentGrid->npf;

  // Find the index of the current subgrid.
  auto x_mirror_glob = Kokkos::create_mirror_view(Kokkos::HostSpace(), this->x_glob[direction]);
  auto x_mirror = Kokkos::create_mirror_view(Kokkos::HostSpace(), this->x[direction]);
  Kokkos::deep_copy(x_mirror_glob,this->x_glob[direction]);
  Kokkos::deep_copy(x_mirror,this->x[direction]);
  int iref = -1;
  for(int i = 0 ; i < x_mirror.extent(0) - 1 ; i++) {
    if((x_mirror(i+1) > x0) && (x_mirror(i) <= x0)) {
      if(x_mirror(i+1) - x0 > x0 - x_mirror(i) ) {
        // we're closer to x(i)
        iref = i;
      } else {
        iref = i+1;
      }
      break;
    }
  }
  if(iref < 0) {
    // x0 is outside of the current rank's grid
    containsX0 = false;
    this->index = 0; // we set the index to 0 by default, but it won't be used since containsX0 is false
  } else {
    containsX0 = true;
    this->index = iref;
  }

  iref = -1;
  for(int i = 0 ; i < x_mirror_glob.extent(0) - 1 ; i++) {
    if((x_mirror_glob(i+1) > x0) && (x_mirror_glob(i) <= x0)) {
      if(x_mirror_glob(i+1) - x0 > x0 - x_mirror_glob(i) ) {
        // we're closer to x(i)
        iref = i;
      } else {
        iref = i+1;
      }
      break;
    }
  }
  if(iref < 0) {
    // x0 is outside of the global grid, this should not happen if it was not outside of the local grid
    throw std::runtime_error("x0 is outside of the parent grid");
  }
  this->index_glob = iref;
  this->x0 = x_mirror_glob(iref);
  // Slice the MPI communicator
  #ifdef WITH_MPI
    int color = containsX0 ? 0 : 1; // processes that contain x0 are in color 0, others in color 1  
    MPI_Comm_split(parentGrid->comm, color, 0, &this->comm);
    MPI_Comm_rank(this->comm, &this->prank);
  #endif

  // Initialise the sub grid dimension which is sliced.
  auto kx = Array1D<real>("Grid_kx",1);
  auto xorigin = parentGrid->x[dir];
  auto x = Kokkos::subview(parentGrid->x[dir], std::make_pair(this->index, this->index+1));

  // Replace the arrays in the current subgrid
  this->x[dir] = x;
  this->kx[dir] = kx;
  this->xbeg_glob[dir] = x0;
  this->xend_glob[dir] = x0 + this->dx[dir];
  this->xbeg[dir] = x0;
  this->xend[dir] = x0 + this->dx[dir];

  this->npr_glob[dir] = 1;
  this->npf_glob[dir] = 1;
  this->npr[dir] = 1;
  this->npf[dir] = 1;


}