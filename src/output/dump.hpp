// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef OUTPUT_DUMP_HPP_
#define OUTPUT_DUMP_HPP_

#include <map>
#include <vector>
#include <cstdio>
#include <string>
#include "field.hpp"
#include "global.hpp"
#ifdef WITH_MPI
#include <mpi.h>
#endif
#if __has_include(<filesystem>)
  #include <filesystem> // NOLINT [build/C++20]
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

// File handler depends on the type of I/O we use
#ifdef WITH_MPI
using DumpFileHandler = MPI_File;
#else
using DumpFileHandler = FILE*;
#endif

class Grid;

class Dump {
 public:
  Dump(Grid *grid, std::string filename);

  void Register(std::string name, Field<Array3D<complex>>);
  void Register(std::string name, real);
  void Register(std::string name, int);
  void Write();

  void Fetch(std::string name, Field<Array3D<complex>> &);
  void Fetch(std::string name, real &);
  void Fetch(std::string name, int &);
  void Read();

  void ReadSnoopy(); // Read snoopy dump files

 private:
  enum DataType {DoubleType, SingleType,
                ComplexDoubleType, ComplexSingleType,
                IntegerType, BoolType};
  void WriteString(DumpFileHandler file, std::string str);
  std::string ReadString(DumpFileHandler file);

  template <typename T>
  void WriteData(DumpFileHandler fileHdl, std::string name, T data);

  template <typename T>
  void ReadData(DumpFileHandler fileHdl, T &data);

  template <typename T>
  void ReadSnoopyScalar(DumpFileHandler fileHdl, T &data);
  template <typename T>
  Dump::DataType TypeToInt();

  void ReadSnoopyArray(DumpFileHandler fileHdl, std::string name, Field<Array3D<complex>> &data);


  void ReadNextFieldProperties(DumpFileHandler fileHdl, std::vector<int> &dim,
                                        Dump::DataType &type, std::string &name);



  static constexpr int StringMaxLength{64};
  bool isRoot{false};
  std::array<int,3> npf_glob, npf, npr_glob;
  std::map<std::string, Field<Array3D<complex>>> fields;
  std::map<std::string, real> reals;
  std::map<std::string, int> ints;

  fs::path filename;
  #ifdef WITH_MPI
    MPI_Offset offset;
    MPI_Datatype view;
    MPI_Comm comm;
  #endif
};

// Implementation of template functions

// Check if type is a fundamental type for dump I/O
template<typename T>
struct IsFundamentalType { enum { result = false }; };

// Template specializations for each fundamental type
template<>
struct IsFundamentalType<int> { enum { result = true }; };

template<>
struct IsFundamentalType<double> { enum { result = true }; };

template<>
struct IsFundamentalType<float> { enum { result = true }; };

template<>
struct IsFundamentalType<complex> { enum { result = true }; };

template <typename T>
  Dump::DataType Dump::TypeToInt() {
    if constexpr (std::is_same<T, double>::value) {
      return Dump::DoubleType;
    } else if constexpr (std::is_same<T, float>::value) {
      return Dump::SingleType;
    } else if constexpr (std::is_same<T, Kokkos::complex<double>>::value) {
      return Dump::ComplexDoubleType;
    } else if constexpr (std::is_same<T, Kokkos::complex<float>>::value) {
      return Dump::ComplexSingleType;
    } else if constexpr (std::is_same<T, int>::value) {
      return Dump::IntegerType;
    } else if constexpr (std::is_same<T, bool>::value) {
      return Dump::BoolType;
    }
    throw std::runtime_error("Unsupported type for TypeToInt");
    return Dump::BoolType; // To suppress compiler warning
  }


template <typename T>
void Dump::WriteData(DumpFileHandler fileHdl, std::string name, T data) {
  astra::pushRegion("Dump::WriteData");
  int type,ndim;
  size_t ntot,size,nglob;

  std::vector<int> dims = {1};
  if constexpr(std::is_same<T, ArrayHost3D<complex>>::value) {
    ntot = data.extent(0);
    ntot *= data.extent(1);
    ntot *= data.extent(2);
    ndim = 3;
    type = static_cast<int>(TypeToInt<complex>());
    size = sizeof(complex);
    nglob = static_cast<size_t>(npf_glob[0])*
            static_cast<size_t>(npf_glob[1])*
            static_cast<size_t>(npf_glob[2]);
    // Overwrite dims
    dims = {static_cast<int>(npf_glob[0]),
            static_cast<int>(npf_glob[1]),
            static_cast<int>(npf_glob[2])};
  } else {
    ntot = 1;   // Number of elements to be written
    nglob = 1;
    size = sizeof(T);
    type = static_cast<int>(TypeToInt<T>());
    ndim = 1;
  }

  // Write field name
  WriteString(fileHdl, name);

  #ifdef WITH_MPI
    MPI_Status status;
    MPI_Datatype MpiType;

    // Write data type
    MPI_File_set_view(fileHdl, offset, MPI_BYTE,
                                    MPI_CHAR, "native", MPI_INFO_NULL );
    if(isRoot) {
        MPI_File_write(fileHdl, &type, 1, MPI_INT, &status);
    }
    offset=offset+sizeof(int);

    // Write dimensions
    MPI_File_set_view(fileHdl, offset, MPI_BYTE,
                                    MPI_CHAR, "native", MPI_INFO_NULL );
    if(isRoot) {
      MPI_File_write(fileHdl, &ndim, 1, MPI_INT, &status);
    }
    offset=offset+sizeof(int);

    for(int n = 0 ; n < ndim ; n++) {
      MPI_File_set_view(fileHdl, offset, MPI_BYTE,
                                      MPI_CHAR, "native", MPI_INFO_NULL );
      if(isRoot) {
        MPI_File_write(fileHdl, &dims[n], 1, MPI_INT, &status);
      }
      offset=offset+sizeof(int);
    }

    if constexpr(std::is_same<T, ArrayHost3D<complex>>::value) {
      MPI_File_set_view(fileHdl, offset, MPI_C_DOUBLE_COMPLEX,  this->view, "native", MPI_INFO_NULL );
      MPI_File_write_all(fileHdl, data.data(), ntot, MPI_C_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);
    } else {
      static_assert(IsFundamentalType<T>::result, "Unsupported type for WriteData");
      // Write raw data
      MPI_File_set_view(fileHdl, offset, MPI_BYTE,
                                      MPI_CHAR, "native", MPI_INFO_NULL );
      if(isRoot) {
        MPI_File_write(fileHdl, &data, ntot*size, MPI_BYTE, &status);
      }
    }
    // increment offset accordingly
    offset += nglob*size;

  #else
    // Write type of data
    if(fwrite(&type, sizeof(int), 1, fileHdl) != 1) {
      throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
    }
    // Write dimensions of array
    if(fwrite(&ndim, sizeof(int), 1, fileHdl) != 1) {
      throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
    }
    if(fwrite(dims.data(), sizeof(int), ndim, fileHdl) !=  static_cast<size_t>(ndim)) {
      throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
    }

    if constexpr(std::is_same<T, ArrayHost3D<complex>>::value) {
      if(fwrite(data.data(), size, ntot, fileHdl) != ntot) {
        throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
      }
    } else {
      static_assert(IsFundamentalType<T>::result, "Unsupported type for WriteData");
      // Write raw data
      if(fwrite(&data, size, ntot, fileHdl) != ntot) {
        throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
      }
    }
  #endif
  astra::popRegion();
}

template <typename T>
void Dump::ReadData(DumpFileHandler fileHdl, T& data) {
  astra::pushRegion("Dump::ReadData");
  size_t ntot,size,nglob;
  if constexpr(std::is_same<T, ArrayHost3D<complex>>::value) {
    ntot = data.extent(0);
    ntot *= data.extent(1);
    ntot *= data.extent(2);
    size = sizeof(complex);
    nglob = static_cast<size_t>(npf_glob[0])*
            static_cast<size_t>(npf_glob[1])*
            static_cast<size_t>(npf_glob[2]);
  } else {
    size = sizeof(T);
    ntot = 1;
    nglob = 1;
  }

  #ifdef WITH_MPI
    MPI_Status status;
    if constexpr(std::is_same<T, ArrayHost3D<complex>>::value) {
      MPI_File_set_view(fileHdl, offset, MPI_C_DOUBLE_COMPLEX, this->view, "native", MPI_INFO_NULL );
      MPI_File_read_all(fileHdl, data.data(), ntot, MPI_C_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);

    } else {
      MPI_File_set_view(fileHdl, this->offset, MPI_BYTE,
                                      MPI_CHAR, "native", MPI_INFO_NULL );
      if(isRoot) {
        MPI_File_read(fileHdl, &data, ntot*size, MPI_BYTE, &status);
      }
      MPI_Bcast(&data, ntot*size, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    offset+= nglob*size;
  #else
    if constexpr(std::is_same<T, ArrayHost3D<complex>>::value) {
      if(fread(data.data(), size, ntot, fileHdl) < ntot) {
        throw std::runtime_error("Error: unexpected end of dump file");
      }
     } else {
      // Read raw data
      if(fread(&data, size, ntot, fileHdl) < ntot) {
        throw std::runtime_error("Error: unexpected end of dump file");
      }
    }
  #endif
  // Check that data was read correctly

  if constexpr(std::is_same<T, ArrayHost3D<complex>>::value) {
    for(int i=0 ; i < data.extent(0) ; i++) {
      for(int j=0 ; j < data.extent(1) ; j++) {
        for(int k=0 ; k < data.extent(2) ; k++) {
          if(std::isnan(data(i,j,k).real()) || std::isnan(data(i,j,k).imag())) {
            throw std::runtime_error("Error: NaN value encountered when reading dump file at (i,j,k)=(" +
                                     std::to_string(i) + "," +
                                     std::to_string(j) + "," +
                                     std::to_string(k) + ")");
          }
        }
      }
    }
  } else {
    if(std::isnan(data)) {
      throw std::runtime_error("Error: NaN value encountered when reading dump file");
    }
  }
  astra::popRegion();
}


template <typename T>
void Dump::ReadSnoopyScalar(DumpFileHandler fileHdl, T &data) {
  const int size = sizeof(T);
  #ifdef WITH_MPI
    MPI_Status status;

    MPI_File_set_view(fileHdl, this->offset, MPI_BYTE,
                                      MPI_CHAR, "native", MPI_INFO_NULL );
    if(isRoot) {
      MPI_File_read(fileHdl, &data, size, MPI_BYTE, &status);
    }
    MPI_Bcast(&data, size, MPI_BYTE, 0, MPI_COMM_WORLD);
    offset+= size;
  #else

  // Read raw data
  if(fread(&data, size, 1, fileHdl) < 1) {
    throw std::runtime_error("Error: unexpected end of dump file");
  }
  #endif
}
#endif// OUTPUT_DUMP_HPP_
