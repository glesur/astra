// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************


// A generic interface for statistics

#ifndef OUTPUT_TIMEVAR_TIMEVAR_HPP_
#define OUTPUT_TIMEVAR_TIMEVAR_HPP_


#include <limits>
#include <string>
#include <vector>
#if __has_include(<filesystem>)
  #include <filesystem> // NOLINT [build/C++20]
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include "field.hpp"
#include "input.hpp"

class Grid;

class TimeVar {
 public:
  TimeVar(Input &input, Grid *grid, std::string name, std::string directory);
  virtual ~TimeVar() {}
  virtual void Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) = 0;
  void Reset();
  std::string GetName() const { return name; }


 protected:
  Grid *grid;
  std::string name;
  fs::path filename;
  std::string directory;
  bool isRoot{false};
  std::string ExtractVarName(const std::string& name);
};


class TimeVarOutput {
 public:
  TimeVarOutput(Input &input, Grid *grid);
  void Write(Field<Array3D<complex>>& field, const real t);
  void Reset();

 private:
  // Hashing functions
  static uint64_t str2int(const std::string& str) {
    return str2int(str.c_str());
  }

  static constexpr int64_t str2int(const char* str, int h = 0) {
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
  }

  Grid *grid;
  std::vector<std::unique_ptr<TimeVar>> timevarList;
};

#endif // OUTPUT_TIMEVAR_TIMEVAR_HPP_
