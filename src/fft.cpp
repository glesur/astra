// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <KokkosFFT.hpp>
#include <Kokkos_Random.hpp>

#include "fft.hpp"
#include "global.hpp"
#include "loop.hpp"
#include "transpose.hpp"

template <typename T> void ShowExtent(T array) {
  for(int i = 0 ; i < 3 ; i++) {
    astra::cout << array.extent(i) << "  ";
  }
  astra::cout << std::endl;
}

  // Empty constructor
FFT::FFT() {};

FFT::FFT(std::array<int,3> npr_glob, std::array<int,3> npf_glob) {
    this->npf_glob = npf_glob;
    this->npr_glob = npr_glob;
    // Local dimensions
    // We assume a decomposition along the first dimension only for now
    this->npr = npr_glob;
    this->npf = npf_glob;
    this->npr_t = npr_glob;

    #ifdef WITH_MPI
      this->npr[0] = npr_glob[0]/astra::psize;
      this->npf[0] = npf_glob[0]/astra::psize;
      this->npr_t[0] = npr_glob[1]/astra::psize;
      this->npr_t[1] = npr_glob[0];
    #endif

    tempReal = Array3D<real>("FFT temp real", npr);
    tempComplex = Array3D<complex>("FFT temp complex", npf);

    // Create the FFT plans
    this->r2cPlan = std::make_unique<PlanR2CType>(Kokkos::DefaultExecutionSpace(), tempReal, tempComplex, KokkosFFT::Direction::forward, std::array<int,3>{-3,-2,-1});
    this->c2rPlan = std::make_unique<PlanC2RType>(Kokkos::DefaultExecutionSpace(), tempComplex, tempReal, KokkosFFT::Direction::backward, std::array<int,3>{-3,-2,-1});
    
    #ifdef WITH_MPI
      // Allocate temporary arrays for domain-splited FFTs and transposes
      this->tempTransposedComplex = Array3D<complex>("FFT transpose temp", npf[1]/astra::psize, npf[0]*astra::psize, npf[2]);
      this->tempTransposedComplex2 = Array3D<complex>("FFT transpose temp2", npf[1]/astra::psize, npf[0]*astra::psize, npf[2]);
      this->tempTransposedReal = Array3D<real>("FFT transpose temp2", npr[1]/astra::psize, npr[0]*astra::psize, npr[2]);
      Array3D<complex> tempComplex2("FFT temp complex", npf);

     // MPI C2R Plans
      // Axis 1 transposed is axis2 for the fft library
      this->c2ciMPIPlan_axis2 = std::make_unique<PlanC2CType1D>(Kokkos::DefaultExecutionSpace(), tempComplex, tempComplex2, KokkosFFT::Direction::backward, std::array<int,1>{-2});
      this->c2rMPIPlan_axis1t3 = std::make_unique<PlanC2RType2D>(Kokkos::DefaultExecutionSpace(), tempTransposedComplex, tempTransposedReal, KokkosFFT::Direction::backward, std::array<int,2>{-2,-1});
      
      // MPI R2C plans
      this->r2cMPIPlan_axis1t3 = std::make_unique<PlanR2CType2D>(Kokkos::DefaultExecutionSpace(), tempTransposedReal, tempTransposedComplex, KokkosFFT::Direction::forward, std::array<int,2>{-2,-1});      
      this->c2cfMPIPlan_axis2 = std::make_unique<PlanC2CType1D>(Kokkos::DefaultExecutionSpace(), tempComplex, tempComplex2, KokkosFFT::Direction::forward, std::array<int,1>{-2});
    
      this->transposeComplex = std::make_unique<Transpose<complex>>(npf);
      this->transposeReal = std::make_unique<Transpose<real>>(npr);

      #endif
    havePlan = true;
  };

// Perform a real-to-complex FFT
void FFT::R2C(const Array3D<real>& in, Array3D<complex>& out, bool transpose) {
  astra::pushRegion("FFT::R2C");

  #ifdef WITH_MPI
    R2C_MPI(in, out, transpose);
  #else
      // Ensure that in array is not erased
    Kokkos::deep_copy(tempReal, in);
    if(havePlan) {
      KokkosFFT::execute(*(r2cPlan.get()), tempReal, out);
    } else {
      KokkosFFT::rfftn(Kokkos::DefaultExecutionSpace(), tempReal, out);
    }
  #endif
  astra::popRegion();
}

void FFT::R2C_MPI(const Array3D<real>& in, Array3D<complex>& out, bool transpose) {
  astra::pushRegion("FFT::R2C_MPI");
  
  if(transpose) {
    this->transposeReal->Apply(in,tempTransposedReal);
    KokkosFFT::execute(*(r2cMPIPlan_axis1t3.get()), tempTransposedReal, tempTransposedComplex);
  } else {
    KokkosFFT::execute(*(r2cMPIPlan_axis1t3.get()), in, tempTransposedComplex);
  }
  this->transposeComplex->Apply(tempTransposedComplex,tempComplex);
  KokkosFFT::execute(*(c2cfMPIPlan_axis2.get()), tempComplex, out);

  astra::popRegion();
}

// Perform a complex-to-real inverse FFT
void FFT::C2R(const Array3D<complex>& in, Array3D<real>& out, bool transpose) {
  astra::pushRegion("FFT::C2R");
    #ifdef WITH_MPI
      C2R_MPI(in,out,transpose);
    #else
      // Ensure that in array is not erased
      Kokkos::deep_copy(tempComplex, in);
      if(havePlan) {
        KokkosFFT::execute(*(c2rPlan.get()), tempComplex, out);
      } else {
        KokkosFFT::irfftn(Kokkos::DefaultExecutionSpace(), tempComplex, out);
      }
    #endif

  astra::popRegion();
}

void FFT::C2R_MPI(const Array3D<complex>& in, Array3D<real>& out, bool transpose) {
  astra::pushRegion("FFT::C2R_MPI");
  KokkosFFT::execute(*(c2ciMPIPlan_axis2.get()), in, tempComplex);
  this->transposeComplex->Apply(tempComplex,tempTransposedComplex);
  if(transpose) {
    KokkosFFT::execute(*(c2rMPIPlan_axis1t3.get()), tempTransposedComplex, tempTransposedReal);
    this->transposeReal->Apply(tempTransposedReal,out);    
  } else {
    KokkosFFT::execute(*(c2rMPIPlan_axis1t3.get()), tempTransposedComplex, out);
  }
  astra::popRegion();
}

// FFT on Host, using the device.
void FFT::R2C_Host(const ArrayHost3D<real>& in, ArrayHost3D<complex>& out) {
  astra::pushRegion("FFT::R2C_Host");
  Array3D<real> inDev = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), in);
  Array3D<complex> outDev = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), out);
  this->R2C(inDev, outDev);
  Kokkos::deep_copy(out, outDev);
  astra::popRegion();
}

void FFT::C2R_Host(const ArrayHost3D<complex>& in, ArrayHost3D<real>& out) {
  astra::pushRegion("FFT::C2R_Host");
  Array3D<complex> inDev = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), in);
  Array3D<real> outDev = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), out);
  this->C2R(inDev, outDev);
  Kokkos::deep_copy(out, outDev);
  astra::popRegion();
}


void FFT::TestMPI() {
#ifdef WITH_MPI
  if(npr_glob[0] % astra::psize != 0 || npr_glob[1] % astra::psize != 0) {
    throw std::runtime_error("Global problem size must be dividible by the number of MPI processes");
  }

  // Make an array with the local problem size
  std::array<int,3> npr = npr_glob;
  npr[0] /= astra::psize;
  std::array<int,3> npf = npr;
  npf[2] = npr[2]/2+1;
  std::array<int,3> npf_glob = npr_glob;
  npf_glob[2] = npr_glob[2]/2+1;

  Kokkos::View<real***, Kokkos::LayoutRight, Device> localReal_right("local real array", npr);
  Array3D<real> localReal("local real array", npr);

  Kokkos::View<real***, Kokkos::LayoutRight, Device> globalReal_right("global real array", npr_glob);
  Array3D<real> globalReal("global real array", npr_glob);

  // Compute a dummy real array
  if(astra::prank == 0) {
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(globalReal_right, random_pool, 1);
  }
  // Scatter to make local arrays
  // And broadcast the global the array
  MPI_Scatter(globalReal_right.data(), npr[0]*npr[1]*npr[2],
              MPI_Astra_real,
              localReal_right.data(), npr[0]*npr[1]*npr[2],
              MPI_Astra_real,
              0, MPI_COMM_WORLD);

  MPI_Bcast(globalReal_right.data(), npr_glob[0]*npr_glob[1]*npr_glob[2],
            MPI_Astra_real, 0, MPI_COMM_WORLD);
  
  // Change array Layout
  astra_for("Reshape global",0, npr_glob[0],
                            0, npr_glob[1],
                            0, npr_glob[2],
    KOKKOS_LAMBDA(int i, int j, int k) {
      globalReal(i,j,k) = globalReal_right(i,j,k);
    });

  // Change array Layout
  astra_for("Reshape local",0, npr[0],
                            0, npr[1],
                            0, npr[2],
    KOKKOS_LAMBDA(int i, int j, int k) {
      localReal(i,j,k) = localReal_right(i,j,k);
    });

    // Create the complex arrays
    Array3D<complex> localComplex("local complex array", npf);
    Array3D<complex> globalComplex("global complex array", npf_glob);

    // Compute the full serial fft
    KokkosFFT::rfftn(Kokkos::DefaultExecutionSpace(), globalReal, globalComplex);

    // Compute the parallele fft
    this->R2C_MPI(localReal, localComplex);

    // Check on a per-process that they all agree
    bool error = false;
    int offset = npf[0]*astra::prank;

    astra_for("Reshape local",0, npf[0],
                            0, npf[1],
                            0, npf[2],
    KOKKOS_LAMBDA(int i, int j, int k) {
      int iglob = i+offset;
      real norm = std::pow(localComplex(i,j,k).real()-globalComplex(iglob,j,k).real(),2)
                  +std::pow(localComplex(i,j,k).real()-globalComplex(iglob,j,k).real(),2);

      norm=std::sqrt(norm);

      if(norm > 1e-8) {
        Kokkos::abort("incoherent values after MPI ifft");
      }
    });

    // Compute the parallele fft
    this->C2R_MPI(localComplex, localReal);

    // Check on a per-process that they all agree
    offset = npr[0]*astra::prank;

    astra_for("Reshape local",0, npr[0],
                            0, npr[1],
                            0, npr[2],
    KOKKOS_LAMBDA(int i, int j, int k) {
      int iglob = i+offset;
      real norm = std::fabs(localReal(i,j,k)-globalReal(iglob,j,k));

      if(norm > 1e-8) {
        Kokkos::abort("incoherent values after MPI ifft");
      }
    });
#endif
}
