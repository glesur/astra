// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef GRIDHOST_HPP_
#define GRIDHOST_HPP_
#include <vector>

#include "astra.hpp"
#include "grid.hpp"
#include "arrays.hpp"


///////////////////////////////////////////
/// A minimal host image of the grid class
///////////////////////////////////////////

class GridHost : Grid {
 public:
  // Constructor from a Grid
  explicit GridHost(Grid &grid) : Grid(grid) {
    // Create host mirror of the grid arrays
    for(int dir = 0 ; dir < 3 ; dir++) {
      x_glob[dir] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.x_glob[dir]);
      x[dir] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.x[dir]);
      kx_glob[dir] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.kx_glob[dir]);
      kx[dir] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.kx[dir]);
    }
  }
  // Type masking of the device arrays
  std::array<ArrayHost1D<real>,3> x_glob;    ///< geometrical central points
  std::array<ArrayHost1D<real>,3> x;         ///< local geometrical central points

  std::array<ArrayHost1D<real>,3> kx_glob;    ///< wavenumbers global
  std::array<ArrayHost1D<real>,3> kx;         ///< wavenumbers local
};

#endif // GRIDHOST_HPP_
