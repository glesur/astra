// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef SHEAR_NOSHEAR_HPP_
#define SHEAR_NOSHEAR_HPP_

#include "astra.hpp"
#include "input.hpp"
#include "arrays.hpp"

class NoShear {
 public:
  real shearRate{0.0};
  NoShear() = default;
  ~NoShear() = default;
  NoShear(Input &input) {};

  void Refresh(real time) {
    // Nothing to do
  }

  void Remap(real time, Array3D<complex>& field) {
    // Nothing to do
  }
  KOKKOS_INLINE_FUNCTION real kx1t(real kx1, real kx2, real kx3) const {
    return kx1;
  }

  KOKKOS_INLINE_FUNCTION real kx2t(real kx1, real kx2, real kx3) const {
    return kx2;
  }

  KOKKOS_INLINE_FUNCTION real kx3t(real kx1, real kx2, real kx3) const {
    return kx3;
  }
};

#endif // SHEAR_NOSHEAR_HPP_