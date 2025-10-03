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
#include "mpi.hpp"
// A class that wraps KokkosFFT functionality
using PlanR2CType = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, Array3D<real>, Array3D<complex>,3>;
using PlanC2RType = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, Array3D<complex>, Array3D<real>,3>;

using PlanC2CType1D = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, Array3D<complex>, Array3D<complex>,1>;
using PlanR2CType2D = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, Array3D<real>, Array3D<complex>,2>;
using PlanC2RType1D = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, Array3D<complex>, Array3D<real>,1>;
using PlanR2CType1D = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace, Array3D<real>, Array3D<complex>,1>;


class FFT {
public:
  // Empty constructor
  FFT() {};

  FFT(std::array<int,3> npr, std::array<int,3> npf);

  // Perform a real-to-complex FFT
  void R2C(const Array3D<real>& in, Array3D<complex>& out);
  void R2C_MPI(const Array3D<real>& in, Array3D<complex>& out);

  // Perform a complex-to-real inverse FFT
  void C2R(const Array3D<complex>& in, Array3D<real>& out);
  void C2R_MPI(const Array3D<complex>& in, Array3D<real>& out);

  void Transpose(const Array3D<complex>& in, Array3D<complex>& out);

  // FFT on Host, using the device.
  void R2C_Host(const ArrayHost3D<real>& in, ArrayHost3D<complex>& out);
  void C2R_Host(const ArrayHost3D<complex>& in, ArrayHost3D<real>& out);
  void TestTranspose();
  void TestMPI();
private:
  bool havePlan{false};
  std::unique_ptr<PlanR2CType> r2cPlan;
  std::unique_ptr<PlanC2RType> c2rPlan;

  // MPI plans
  // Backard (C2R)
  std::unique_ptr<PlanC2RType1D> c2rMPIPlan_axis3;
  std::unique_ptr<PlanC2CType1D> c2ciMPIPlan_axis1t;
  std::unique_ptr<PlanC2CType1D> c2ciMPIPlan_axis2;

  // Forward (R2C)
  std::unique_ptr<PlanR2CType2D> r2cMPIPlan_axis23;
  std::unique_ptr<PlanC2CType1D> c2cfMPIPlan_axis1t;
  
  // Temporary arrays for MPI FFTs
  Array3D<complex> tempComplex;
  Array3D<complex> tempTransposedComplex;
  Array3D<complex> tempTransposedComplex2;

  Array3D<complex> tempXY;
  Array3D<complex> tempYX;
  std::array<int,3> npr; // Local real space dimensions
  std::array<int,3> npr_t; // Local real space dimensions after transpose
  std::array<int,3> npf; // Local fourier space dimensions
  std::array<int,3> npf_glob; // Global fourier space dimensions
  std::array<int,3> npr_glob; // Global real space dimensions
};

#endif // FFT_HPP_