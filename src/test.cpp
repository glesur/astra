// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include "test.hpp"
#include "astra.hpp"
#include "input.hpp"
#include "fft.hpp"
#include "transpose.hpp"

Test::Test(Input *input) {
  this->input = input;
}

void Test::Run() {
  astra::cout << "Tests: Running tests..." << std::endl;
  int n = 16;
  std::array<int,3> nr = {n*astra::psize, n, n};
  std::array<int,3> nf = {n*astra::psize, n, n/2+1};

  FFT fft(nr, nf);
  #ifdef WITH_MPI
    astra::cout << "Test: Testing complex Transpose" << std::endl;
    Transpose<complex> transpose_c(nf); // NB: ng is not used for tests
    transpose_c.Test();
    astra::cout << "Test: Testing real Transpose" << std::endl;
    Transpose<real> transpose_r(nr);
    transpose_r.Test();
    astra::cout << "Test: Testing MPI distributed FFT" << std::endl;
    fft.TestMPI();
  #endif
}

void Test::Finalize() {
  astra::cout << "Test: All tests passed!" << std::endl;
  Kokkos::finalize();
  #ifdef WITH_MPI
    MPI_Finalize();
  #endif
  exit(0);
}
