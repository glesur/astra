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
using PlanR2CType = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, Array3D<real>, Array3D<complex>,3>;
using PlanC2RType = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, Array3D<complex>, Array3D<real>,3>;

class FFT {
public:
  // Empty constructor
  FFT() {};

  FFT(std::array<int,3> npr, std::array<int,3> npf);

  // Perform a real-to-complex FFT
  void R2C(const Array3D<real>& in, Array3D<complex>& out);

  // Perform a complex-to-real inverse FFT
  void C2R(const Array3D<complex>& in, Array3D<real>& out);

  // FFT on Host, using the device.
  void R2C_Host(const ArrayHost3D<real>& in, ArrayHost3D<complex>& out);
  void C2R_Host(const ArrayHost3D<complex>& in, ArrayHost3D<real>& out);
private:
  bool havePlan{false};
  std::unique_ptr<PlanR2CType> r2cPlan;
  std::unique_ptr<PlanC2RType> c2rPlan;
};

#endif // FFT_HPP_