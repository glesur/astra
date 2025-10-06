// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef OUTPUT_VTK_HPP_
#define OUTPUT_VTK_HPP_


#include <string>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#if __has_include(<filesystem>)
  #include <filesystem> // NOLINT [build/c++17]
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include "input.hpp"
#include "arrays.hpp"
#include "field.hpp"
#include "bigEndian.hpp"

#ifdef WITH_MPI
#include <mpi.h>
#endif


// File handler depends on the type of I/O we use
#ifdef WITH_MPI
using VtkFileHandler = MPI_File;
#else
using VtkFileHandler = FILE*;
#endif

class Grid;

class Vtk {

 public:
  Vtk(Grid *grid, real time, std::string filename = "data", std::string outputDirectory = "./");   // init VTK object
  void Write(Field<Array3D<complex>> field);     // Write content of a field class
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

  void WriteHeader(VtkFileHandler, real);
  void WriteScalar(VtkFileHandler, float*,  const std::string &);
  void WriteHeaderString(const char* header, VtkFileHandler fvtk);
  void WriteHeaderBinary(float *buffer, int64_t nelem, VtkFileHandler fvtk); 

};


#endif // OUTPUT_VTK_HPP_