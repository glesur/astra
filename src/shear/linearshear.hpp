// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef SHEAR_LINEARSHEAR_HPP_
#define SHEAR_LINEARSHEAR_HPP_

#include <KokkosFFT.hpp>
#include "astra.hpp"
#include "input.hpp"
#include "arrays.hpp"
#include "grid.hpp"
#include "loop.hpp"
#include "global.hpp"
#ifdef WITH_MPI
#include "transpose.hpp"
#endif
#include "noshear.hpp"

class LinearShear : public NoShear {
 public:
  static constexpr bool isEnabled{true};
  LinearShear() = default;
  ~LinearShear() = default;
  LinearShear(Input &input, Grid* grid) : NoShear(input,grid) {
    this->shearRate = input.Get<real>("Physics","shear_rate",0);
    lx = grid->xend_glob[IDIR] - grid->xbeg_glob[IDIR];
    ly = grid->xend_glob[JDIR] - grid->xbeg_glob[JDIR];
    x0 = grid->xbeg_glob[IDIR];
    remapInterval = ly / (std::fabs(this->shearRate) * lx);
    this->grid = grid;
  }

  void Refresh(real t) {
    tremap = t - nremaps * remapInterval;
    x0advection = std::fmod(t*this->shearRate*x0, ly);
  }

  void SetTinit(real t0) {
    // Compute the number of remaps already performed at t0
    nremaps = static_cast<int>(0.5 + t0/remapInterval);
    this->Refresh(t0);
  }

  bool NeedRemap(real time) {
    return (time - remapInterval * nremaps) >= remapInterval/2;
  }

  void Remap(real time, Array3D<complex>& field) {
    astra::pushRegion("LinearShear::Remap");

    int n = static_cast<int>(0.5 + time/remapInterval) - nremaps;
    if(n<=0) throw std::runtime_error("LinearShear::Remap called but no remap needed");

    int nx_glob = this->grid->npf_glob[IDIR];
    int ny_glob = this->grid->npf_glob[JDIR];

    real kx1max = this->grid->kmax[IDIR];
    real kx2max = this->grid->kmax[JDIR];
    real kx3max = this->grid->kmax[KDIR];
    auto kx1 = this->grid->kx[IDIR];
    auto kx2 = this->grid->kx[JDIR];
    auto kx3 = this->grid->kx[KDIR];
    #ifdef WITH_MPI
      Array3D<complex> transposed("transposed", this->grid->npf_glob[JDIR]/astra::psize, this->grid->npf_glob[IDIR], this->grid->npf_glob[KDIR]);
      Transpose<complex> transpose(this->grid->npf);
      transpose.Apply(field, transposed);
      Array3D<complex> temp("temp", transposed.extent(0), transposed.extent(1), transposed.extent(2));
      Kokkos::deep_copy(temp, Kokkos::complex(0.0,0.0));
      const int ny_start = this->grid->npf_glob[JDIR]*astra::prank/astra::psize;
      astra_for("shear_remap", 0,transposed.extent(0),0,transposed.extent(1),0,transposed.extent(2),
        KOKKOS_LAMBDA (const int j, const int i, const int k) {
          // unfold Fourier modes
          const int nx = (i+nx_glob/2) % nx_glob - nx_glob/2;
          const int ny = (j+ny_start+ny_glob/2) % ny_glob - ny_glob/2;

          const int nxtarget = nx + n*ny;

          // Check if mode goes out of bounds
          if(nxtarget > -nx_glob/2 && nxtarget <= nx_glob/2) {
            const int inew = (nxtarget + nx_glob) % nx_glob;
            temp(j,inew,k) = transposed(j,i,k);
          }
        }
      );
      transpose.Apply(temp, field);
    #else
      Array3D<complex> temp("temp", field.extent(0), field.extent(1), field.extent(2));
      Kokkos::deep_copy(temp, Kokkos::complex(0.0,0.0));
      astra_for("shear_remap", 0,field.extent(0),0,field.extent(1),0,field.extent(2),
        KOKKOS_LAMBDA (const int i, const int j, const int k) {
          // unfold Fourier modes
          const int nx = (i+nx_glob/2) % nx_glob - nx_glob/2;
          const int ny = (j+ny_glob/2) % ny_glob - ny_glob/2;

          const int nxtarget = nx + n*ny;

          // Check if mode goes out of bounds
          if(nxtarget > -nx_glob/2 && nxtarget <= nx_glob/2) {
            const int inew = (nxtarget + nx_glob) % nx_glob;
            temp(inew,j,k) = field(i,j,k);
          }
        }
      );
      Kokkos::deep_copy(field, temp);
    #endif

    // Apply 2/3 de-aliasing rule
    astra_for("mask_remap", 0,field.extent(0),0,field.extent(1),0,field.extent(2),
      KOKKOS_LAMBDA (const int i, const int j, const int k) {
        complex mask = (std::fabs(kx1(i))< 2./3*kx1max
                      && std::fabs(kx2(j))< 2./3*kx2max
                      && std::fabs(kx3(k))< 2./3*kx3max) ? 1.0 : 0.0;
        field(i,j,k) *= mask;
      }
    );
    // Done
    astra::popRegion();
  }

  void UnshearFrame(Array3D<real> field) {
    Array3D<complex> temp("temp", field.extent(0), field.extent(1), field.extent(2));
    Array3D<complex> temp2("temp2", field.extent(0), field.extent(1), field.extent(2));
    // Forward FFT along x2
    KokkosFFT::fft(Kokkos::DefaultExecutionSpace(), field, temp, KokkosFFT::Normalization::backward, -2);

    auto kx2 = this->grid->kx[JDIR];
    auto x1 = this->grid->x[IDIR];
    real tremap = this->tremap;
    real x0advection = this->x0advection;
    real shearRate = this->shearRate;
    real x0 = this->x0;
    // Shear remap along x1
    astra_for("shear_remap", 0,field.extent(0),0,field.extent(1),0,field.extent(2),
      KOKKOS_LAMBDA (const int64_t i, const int64_t j, const int64_t k) {
        real phase = kx2(j)*(shearRate*tremap*(x1(i)-x0)+x0advection);
        temp(i,j,k) *= Kokkos::complex(cos(phase), sin(phase));
      }
    );
    // Backwards along x2
    KokkosFFT::ifft(Kokkos::DefaultExecutionSpace(), temp, temp2, KokkosFFT::Normalization::backward, -2);
    // Copy real part back to field
    astra_for("copy_real_part", 0,field.extent(0),0,field.extent(1),0,field.extent(2),
      KOKKOS_LAMBDA (const int64_t i, const int64_t j, const int64_t k) {
        field(i,j,k) = temp2(i,j,k).real();
      }
    );
  }


  KOKKOS_INLINE_FUNCTION real kx1t(real kx1, real kx2, real kx3) const {
    return kx1 + this->shearRate * tremap * kx2;
  }

  real kx1tmax() const {
    return this->kx1max + std::fabs(this->shearRate * tremap) * this->kx2max;
  }

  // kx2t and kx3t are unchanged
 private:
  real tremap; // time between +/- remapInterval/2 to compute wave vectors
  real x0advection; // advection of origin due to shear;
  real remapInterval; // time between remaps
  real lx,ly,x0;
  Grid *grid;
  int nremaps{0};
};

#endif // SHEAR_LINEARSHEAR_HPP_
