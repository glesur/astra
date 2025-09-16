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

// Perform a real-to-complex FFT
void FFT::R2C(const Array3D<real>& in, Array3D<complex>& out) {
  astra::pushRegion("FFT::R2C");
  KokkosFFT::rfftn(Kokkos::DefaultExecutionSpace(), in, out);
  astra::popRegion();
}

// Perform a complex-to-real inverse FFT
void FFT::C2R(const Array3D<complex>& in, Array3D<real>& out) {
  astra::pushRegion("FFT::C2R");
  KokkosFFT::irfftn(Kokkos::DefaultExecutionSpace(), in, out);
  astra::popRegion();
}