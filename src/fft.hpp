// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FFT_HPP_
#define FFT_HPP_

#include <KokkosFFT.hpp>
#include "arrays.hpp"
#include "astra.hpp"
// A class that wraps KokkosFFT functionality
class FFT {
public:
  FFT() {};

  // Perform a real-to-complex FFT
  void R2C(const Array3D<real>& in, Array3D<complex>& out);

  // Perform a complex-to-real inverse FFT
  void C2R(const Array3D<complex>& in, Array3D<real>& out);
};

#endif // FFT_HPP_