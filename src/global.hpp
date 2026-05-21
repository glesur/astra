// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// ***********************************************************************************
// Astra MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef GLOBAL_HPP_
#define GLOBAL_HPP_
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include "arrays.hpp"
#include "npy.hpp"

namespace astra {
int initialize();   // Initialisation routine for idefix
real randm();      // Custom random number generator
void safeExit(int );       // Exit the code
class AstraOutStream;
class AstraErrStream;
class Profiler;

extern int prank;                       //< parallel rank
extern int psize;
extern AstraOutStream cout;              //< custom cout for idefix
extern AstraErrStream cerr;              //< custom cerr for idefix
extern Profiler prof;                   //< profiler (for memory & performance usage)
extern LoopPattern defaultLoopPattern;  //< default loop patterns (for idefix_for loops)
extern bool warningsAreErrors;    //< whether warnings should be considered as errors

void pushRegion(const std::string&);
void popRegion();



///< dump Astra array to a numpy array on disk
template<typename ArrayType>
void DumpArray(std::string filename, ArrayType array) {
  auto hArray = Kokkos::create_mirror(array);
  Kokkos::deep_copy(hArray, array);

  std::array<uint64_t, ArrayType::rank> shape;
  bool fortran_order{false};
  for (size_t i = 0; i < ArrayType::rank; ++i) {
    shape[i] = array.extent(i);
  }
  npy::SaveArrayAsNumpy(filename, fortran_order, ArrayType::rank, shape.data(), hArray.data());
}


} // namespace astra

class astra::AstraOutStream {
 public:
  void init(int);
  void enableLogFile();
  // for regular output of variables and stuff
  template<typename T> AstraOutStream& operator<<(const T& something) {
    if(toscreen) std::cout << something;
    if(logFileEnabled) my_fstream << something;
    return *this;
  }
  // for manipulators like std::endl
  typedef std::ostream& (*stream_function)(std::ostream&);
  AstraOutStream& operator<<(stream_function func) {
    if(toscreen) func(std::cout);
    if(logFileEnabled) func(my_fstream);
    return *this;
  }
 private:
  std::ofstream my_fstream;
  bool toscreen;
  bool logFileEnabled{false};   //< whether streams are also written to a log file
};

class astra::AstraErrStream {
 public:
  // for error output of variables and stuff
  template<typename T> AstraErrStream& operator<<(const T& something) {
    std::cerr << something;
    return *this;
  }
  // for manipulators like std::endl
  typedef std::ostream& (*stream_function)(std::ostream&);
  AstraErrStream& operator<<(stream_function func) {
    func(std::cerr);
    return *this;
  }
};


#endif // GLOBAL_HPP_
