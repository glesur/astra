// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef SHEAR_LINEARSHEAR_HPP_
#define SHEAR_LINEARSHEAR_HPP_

#include "astra.hpp"
#include "input.hpp"
#include "arrays.hpp"
#include "grid.hpp"

class LinearShear : public NoShear {
 public:
  real shearRate;
  LinearShear() = default;
  ~LinearShear() = default;
  LinearShear(Input &input, Grid* grid) : NoShear(input,grid) {
    shearRate = input.Get<real>("Physics","shear_rate",0);
    lx = grid->xend_glob[IDIR] - grid->xbeg_glob[IDIR];
    ly = grid->xend_glob[JDIR] - grid->xbeg_glob[JDIR];
    remapTime = ly / (shearRate * lx);
  };
  
  void Refresh(real t) {
    // Nothing to do
    tremap = t - nremaps * remapTime; 
  }

  void SetTinit(real t0) {
    // Compute the number of remaps already performed at t0
    nremaps = static_cast<int>(0.5 + t0/remapTime);
  }

  void Remap(real time, Array3D<complex>& field) {
    // Nothing to do
  }
  KOKKOS_INLINE_FUNCTION real kx1t(real kx1, real kx2, real kx3) const {
    return kx1+ shearRate * tremap * kx2;
  }

  real kx1tmax() const {
    return this->kx1max + std::fabs(shearRate * tremap) * this->kx2max;
  }
  
  // kx2t and kx3t are unchanged
 private:
  real tremap, remapTime;
  real lx,ly;

  int nremaps{0};
};

#endif // SHEAR_LINEARSHEAR_HPP_