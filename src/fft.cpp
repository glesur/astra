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
#include "loop.hpp"

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
  #ifdef WITH_MPI
    Array3D<complex> temp("FFT transpose temp", npf[0], npf[1], npf[2]);
    Array3D<complex> temp_t("FFT transpose temp", npf[1]/astra::psize, npf[0]*astra::psize, npf[2]);
    Array3D<complex> temp_t2("FFT transpose temp", npf[1]/astra::psize, npf[0]*astra::psize, npf[2]);
    Kokkos::rfftn(Kokkos::DefaultExecutionSpace(), in, temp,std::array<int,2>{-3,-2});
    this->Transpose(temp,temp_t);
    Kokkos::fftn(Kokkos::DefaultExecutionSpace(), temp_t, temp_t2,std::array<int,1>{-2});
    this->Transpose(temp_t2,out);
  #else
    if(havePlan) {
      KokkosFFT::execute(*(r2cPlan.get()), in, out);
    } else {
      KokkosFFT::rfftn(Kokkos::DefaultExecutionSpace(), in, out);
    }
  #endif
  astra::popRegion();
}

// Perform a complex-to-real inverse FFT
void FFT::C2R(const Array3D<complex>& in, Array3D<real>& out) {
  astra::pushRegion("FFT::C2R");
  #ifdef WITH_MPI
    Array3D<complex> temp("FFT transpose temp", npf[0], npf[1], npf[2]);
    Array3D<complex> temp_t("FFT transpose temp", npf[1]/astra::psize, npf[0]*astra::psize, npf[2]);
    Array3D<complex> temp_t2("FFT transpose temp", npf[1]/astra::psize, npf[0]*astra::psize, npr[2]);
    Kokkos::ifftn(Kokkos::DefaultExecutionSpace(), in, temp,std::array<int,1>{-2});
    this->Transpose(temp,temp_t);
    Kokkos::ifftn(Kokkos::DefaultExecutionSpace(), temp_t, temp_t2 ,std::array<int,1>{-2});
    this->Transpose(temp_t2,temp);
    Kokkos::irfftn(Kokkos::DefaultExecutionSpace(), temp, out,std::array<int,1>{-3});
  #else
    if(havePlan) {
      KokkosFFT::execute(*(c2rPlan.get()), in, out);
    } else {
      KokkosFFT::irfftn(Kokkos::DefaultExecutionSpace(), in, out);
    }
  #endif
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

/* MPI Transposition routines.
From complex to real, we first transform along x2
Then we transpose x2 and x1 to have x1 contiguous in memory
We then transform along x1 and finally x3 (this last one being a real fft). This gives a final 

Transposition is done as follows:
Star with a matrix A_ijk
we require the first two indices to be dividible by n. so that (i,j) can be divided into n^2
blocks. Hence the indices reads

A_mi'pj'k
where i'=i'/n and j'=j/n while m = i%n and p = j%n
Initially the matrix is distributed accross MPI processes along m,

The first step is to transpose locally pj'=j and i'
B_mpj'i'k = A_mi'pj'k

Then we do an MPI_Alltoall to exchange the data between processors, this transposes m and p
C_pmj'i'k = B_mpj'i'k

Finally, we do a local block transpose between j' and m
D_pj'mi'k = C_pmj'i'k

The Matrix D is the final result with mi' contiguous in memory and pj' distributed accross MPI processes.

Example:
Assuming a full domain of 6x4 elements.

Initial layout contiguous along the second index (x2) and decomposed along the first (x1):
The dots represent the block division inside each MPI process whch is just represented to guide the eye
A11 A12 . A13 A14
A21 A22 . A23 A24
A31 A32 . A33 A34
------------- MPI_Proc division -------------
A41 A42 . A43 A44
A51 A52 . A53 A54
A61 A62 . A63 A64

1st step, local transposition (in each processor):
A11 A21 A31
A12 A22 A32
 .   .   .
A13 A23 A33
A14 A24 A34
------------- MPI_Proc division -------------
A41 A51 A61
A42 A52 A62
 .   .   .
A43 A53 A63
A44 A54 A64


Then we do a MPI_AllToall to exchange the data between processors.
A11 A21 A31
A12 A22 A32
 .   .   .
A41 A51 A61
A42 A52 A62
------------- MPI_Proc division -------------
A13 A23 A33
A14 A24 A34
 .   .   .
A43 A53 A63
A44 A54 A64

Finally the block transpose
A11 A21 A31 . A41 A51 A61
A12 A22 A32 . A42 A52 A62
-------------- MPI_Proc division -------------
A13 A23 A33 . A43 A53 A63
A14 A24 A34 . A44 A54 A64


After local transpose:


From real to complex, we first transform along x3 (real fft) then x1 (which is transposed, so this
is really the second coordinate).
We then transpose x1 and x2 to have x2 contiguous in memory and finally transform along x2.


Then along x3 (real fft)
we first transform along x3 and x2.
Then we need to transpose x1 and x2 to have x1 contiguous in memory.


*/

void FFT::Transpose(const Array3D<complex>& in, Array3D<complex>& out) {
  astra::pushRegion("FFT::Transpose");
  #ifdef WITH_MPI
  const int n = astra::psize; // number of MPI processes
  const int ni = in.extent(0); // Size of local block
  const int nj = in.extent(1)/n;
  const int nk = in.extent(2);
  const int n1 = ni*nj*n; // Full size of the first dimension


  //Array2D<complex> tempB("FFT transpose temp", n1,nk);
  Kokkos::View<complex**, Kokkos::LayoutRight, Device> tempB("FFT transpose temp", n1,nk);
  
  astra_for("FFT::TransposeLocal1", 0,n1, 0, nk,
    KOKKOS_LAMBDA(const int ij, const int k) {
    // for in array, ij = j'+nj*p +i'*nj*n= j+i'*nj*n (given that j=j'+nj*p)
    // For tempB, ij=i'+ni*(j'+nj*p) 
     int i = ij / (nj*n);
     int j = ij- i*nj*n;
     int ijprime = i + ni*j;
     tempB(ijprime, k) = in(i,j,k);
    });
    Kokkos::fence();

  Kokkos::View<complex**, Kokkos::LayoutRight, Device> tempC("FFT transpose temp", n1,nk);
  int ret = MPI_Alltoall(tempB.data(), ni*nj*nk,
                         MPI_Astra_Complex,
                         tempC.data(), ni*nj*nk,
                         MPI_Astra_Complex,
                         MPI_COMM_WORLD);

  astra_for("FFT::TransposeBlock", 0,n1, 0, nk,
    KOKKOS_LAMBDA(const int ij, const int k) {
      // For tempC, ij = i' + ni*(j'+m*nj)
      // For out, ij = i' + m*ni + j'*ni*n (given that i = i'+m*ni)

      const int j = ij / (ni*n);
      const int i = ij - j*ni*n;
      const int m = i / ni;
      const int iprime = i - m*ni;
      const int ijprime = iprime +ni*(j + m*nj);

      out(j,i,k) = tempC(ijprime,k);
    });
    Kokkos::fence();

    #endif
  astra::popRegion();
}

void FFT::TestTranspose() {

  astra::cout << "Testing FFT Transpose" << std::endl;
  const int n = astra::psize; // number of MPI processes
  const int rank = astra::prank; // rank of the current process
  const int ni = 3; // Size of local block
  const int nj = 4;
  const int nk = 9;



  Array3D<complex> in("FFT transpose test input", ni,nj*n,nk);
  Array3D<complex> out("FFT transpose test output", nj,ni*n,nk);

  // Fill the input array with some test values
  astra_for("FFT::TestTransposeFill", 0,ni, 0,nj*n, 0,nk,
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
      in(i,j,k) = k+10*j+100*(i+rank*ni);
    });
  Transpose(in, out);
  ArrayHost3D<complex> outHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);

  // Check the output values
  bool error = false;
  for(int j = 0 ; j < nj ; j++) {
    for(int i = 0 ; i < ni*n ; i++) {
      for(int k = 0 ; k < nk ; k++) {
        if(outHost(j,i,k) != k+10*(j+rank*nj)+100*(i)) {
          astra::cout << "rank " << rank << " Transpose error at (" << i << "," << j << "," << k << "): got " << outHost(j,i,k).real() << " instead of " << k+10*(j+rank*nj)+100*(i) << std::endl;
          error = true;
        }
      }
    }
  }

  if(!error) {
    astra::cout << "Transpose test passed" << std::endl;
  } else {
    throw std::runtime_error("Transpose test failed @ rank " + std::to_string(rank));
  }
}