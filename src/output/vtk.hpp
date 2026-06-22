// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef OUTPUT_VTK_HPP_
#define OUTPUT_VTK_HPP_


#include <string>
#include <cstdio>
#include <memory>
#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#if __has_include(<filesystem>)
  #include <filesystem> // NOLINT [build/C++20]
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "input.hpp"
#include "arrays.hpp"
#include "field.hpp"
#include "bigEndian.hpp"




// File handler depends on the type of I/O we use
#ifdef WITH_MPI
using VtkFileHandler = MPI_File;
#else
using VtkFileHandler = FILE*;
#endif

class Grid;
class LinearShear;

class Vtk {
 public:
  Vtk(Grid *grid, Input &input, real time, std::string filename = "data", std::string outputDirectory = "./");   // init VTK object
  template<typename T> void Write(T array, real time, std::string name);     // Write content of an array class
  template<typename T> void Write(Field<Array3D<T>> field, real time);     // Write content of a field class

  ~Vtk();  // Destructor

 private:
    // Timer
  Kokkos::Timer timer;

  // dimensions
  int64_t nx1,nx2,nx3;
  int64_t nx1loc,nx2loc,nx3loc;
  bool isRoot;  // Whether this process is our root process for the current

  Grid *grid;
  // Array designed to store the temporary vector array
  float *vect3D;

  // File Handler
  VtkFileHandler fileHdl;

  // BigEndian conversion
  BigEndian bigEndian;
  // File offset
#ifdef WITH_MPI
  MPI_Offset offset;
  MPI_Datatype view;
  MPI_Comm comm;
#endif
  bool haveShear{false};
  std::unique_ptr<LinearShear> linearShear;
  void WriteHeader(VtkFileHandler, real);
  void WriteScalar(VtkFileHandler, float*,  const std::string &);
  void WriteHeaderString(const char* header, VtkFileHandler fvtk);
  void WriteHeaderBinary(float *buffer, int64_t nelem, VtkFileHandler fvtk);
};

#include "fft.hpp"
#include "linearshear.hpp"

template<typename T>
void Vtk::Write(T array, real time, std::string name) {
  astra::pushRegion("Vtk::Write");
  auto realHostView = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), array);
  constexpr int rank = decltype(array)::rank();
  // Compute total number of elements in the array
  int64_t nelem = 1;
  for(int i = 0; i < rank; i++) {
    nelem *= array.extent(i);
  }

  for(int64_t idx = 0; idx < nelem; idx++) {
    real q0;
    if constexpr(rank==1) {
      q0 = realHostView(idx);
    } else if constexpr(rank==2) {
      int i = idx % array.extent(0);
      int j = idx / array.extent(0);
      q0 = realHostView(i,j);
    } else if constexpr(rank==3) {
      int i = idx % array.extent(0);
      int j = (idx / array.extent(0)) % array.extent(1);
      int k = idx / (array.extent(0)*array.extent(1));
      q0 = realHostView(i,j,k);
    } else {
      static_assert(rank <= 3, "Unsupported rank for VTK output");
    }
    vect3D[idx] = bigEndian(static_cast<float>(q0));
  }

  WriteScalar(fileHdl, vect3D, name);
  astra::popRegion();
}


template<typename T>
void Vtk::Write(Field<Array3D<T>> field, real time) {
  astra::pushRegion("Vtk::Write");
  ArrayHost3D<real> realHostView("VTK_rfft_host_view",nx1loc,nx2loc,nx3loc);
  Array3D<real> realView;
  // Write field one by one
  for(auto const& [name, view] : field) {
    if constexpr (std::is_same<T, complex>::value) {
      // Fourier transform the view
      realView = Array3D<real>("VTK_rfft_view",nx1loc,nx2loc,nx3loc);
      grid->fft->C2R(view, realView);
      Kokkos::fence();
      // unshear if needed
      if(haveShear) {
        this->linearShear->SetTinit(time);
        this->linearShear->UnshearFrame(realView);
      }
    } else {
      // shallow copy the view that is already real (and assumed unsheared)
      realView = view;
    }
    // Copy data to host
    Kokkos::deep_copy(realHostView, realView);
    for(int k = 0; k < nx3loc ; k++ ) {
      for(int j = 0; j < nx2loc ; j++ ) {
        for(int i = 0; i < nx1loc ; i++ ) {
          vect3D[i + j*nx1loc + k*nx1loc*nx2loc]
              = bigEndian(static_cast<float>(realHostView(i,j,k)));
        }
      }
    }
    WriteScalar(fileHdl, vect3D, name);
  }
  astra::popRegion();
}
#endif // OUTPUT_VTK_HPP_
