// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// ***********************************************************************************
// Astra MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef SUBGRID_HPP_
#define SUBGRID_HPP_
#include <vector>
#include <memory>
#include "arrays.hpp"
#include "input.hpp"
#include "grid.hpp"

enum class SliceType {Cut, Average};

class SubGrid : public Grid {
 public:
  SubGrid(Grid * grid, SliceType type, int dir, real x0);
  SubGrid();
  ~SubGrid();
  Grid *parentGrid; ///< pointer to the parent grid
  int direction; ///< direction of the slice/average
  SliceType sliceType; ///< type of the slice/average (cut or average)
  real x0;   ///< Cell center coordinate of the slice/average
  int index; ///< index in parent grid of the slice/average
  int index_glob; ///< global index in parent grid of the slice/average
  bool containsX0{false};
};

#endif// SUBGRID_HPP_
