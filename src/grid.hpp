// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef GRID_HPP_
#define GRID_HPP_
#include <vector>
#include <memory>

#include "arrays.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////
/// The Grid class is designed to store the grid data of the FULL computational domain on the Host
/// (i.e. of all of the MPI processes running).
/// The domain decomposition is performed by the child instances of the DataBlock class, which are
/// built on a Grid instance, using the cartesian MPI communicator of that Grid. Note that all of
/// the arrays of a Grid are on the device. If a Host access is needed,
/// it is recommended to use a GridHost instance from this Grid, and sync it.
/////////////////////////////////////////////////////////////////////////////////////////////////

class Grid {
 public:
  std::array<Array1D<real>,3> x;    ///< geometrical central points
  std::array<Array1D<real>,3> xr;   ///< cell right interface
  std::array<Array1D<real>,3> xl;   ///< cell left interface
  std::array<real,3> dx;   ///< cell width

  std::array<real,3> xbeg;           ///< Beginning of grid
  std::array<real,3> xend;           ///< End of grid

  std::array<int,3> np;          ///< total number of grid points
  

  // Constructor
  explicit Grid(Input &);
  void ShowConfig();
  Grid() = default;

 private:

};



#endif // GRID_HPP_
