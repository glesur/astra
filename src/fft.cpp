// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <KokkosFFT.hpp>

#include "fft.hpp"
#include "global.hpp"

FFT::FFT(std::array<int,3> npr, std::array<int,3> npf) {
    Array3D<real> tempReal("FFT temp real", npr);
    Array3D<complex> tempComplex("FFT temp complex", npf);
    // Create the FFT plans
    this->r2cPlan = std::make_unique<PlanR2CType>(Kokkos::DefaultExecutionSpace(), tempReal, tempComplex, KokkosFFT::Direction::forward, std::array<int,3>{-3,-2,-1});
    this->c2rPlan = std::make_unique<PlanC2RType>(Kokkos::DefaultExecutionSpace(), tempComplex, tempReal, KokkosFFT::Direction::backward, std::array<int,3>{-3,-2,-1});
    havePlan = true;
  };

// Perform a real-to-complex FFT
void FFT::R2C(const Array3D<real>& in, Array3D<complex>& out) {
  astra::pushRegion("FFT::R2C");
  if(havePlan) {
    KokkosFFT::execute(*(r2cPlan.get()), in, out);
  } else {
    KokkosFFT::rfftn(Kokkos::DefaultExecutionSpace(), in, out);
  }
  astra::popRegion();
}

// Perform a complex-to-real inverse FFT
void FFT::C2R(const Array3D<complex>& in, Array3D<real>& out) {
  astra::pushRegion("FFT::C2R");
  if(havePlan) {
    KokkosFFT::execute(*(c2rPlan.get()), in, out);
  } else {
    KokkosFFT::irfftn(Kokkos::DefaultExecutionSpace(), in, out);
  }
  astra::popRegion();
}

// FFT on Host, using the device.
void FFT::R2C_Host(const ArrayHost3D<real>& in, ArrayHost3D<complex>& out) {
  astra::pushRegion("FFT::R2C_Host");
  Array3D<real> inDev = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), in);
  Array3D<complex> outDev = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), out);
  if(havePlan) {
    KokkosFFT::execute(*(r2cPlan.get()), inDev, outDev);
  } else {
    KokkosFFT::rfftn(Kokkos::DefaultExecutionSpace(), inDev, outDev);
  }
  Kokkos::deep_copy(out, outDev);
  astra::popRegion();
}

void FFT::C2R_Host(const ArrayHost3D<complex>& in, ArrayHost3D<real>& out) {
  astra::pushRegion("FFT::C2R_Host");
  Array3D<complex> inDev = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), in);
  Array3D<real> outDev = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), out);
  if(havePlan) {
    KokkosFFT::execute(*(c2rPlan.get()), inDev, outDev);
  } else {
    KokkosFFT::irfftn(Kokkos::DefaultExecutionSpace(), inDev, outDev);
  }
  Kokkos::deep_copy(out, outDev);
  astra::popRegion();
}