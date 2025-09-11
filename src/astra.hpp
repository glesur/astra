#ifndef ASTRA_HPP_
#define ASTRA_HPP_
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

using Device = Kokkos::DefaultExecutionSpace;
using Layout = Kokkos::LayoutLeft;
using real = double;
using complex = Kokkos::complex<real>;

/// Type of loops we admit in idefix (see loop.hpp for details)
enum class LoopPattern { SIMDFOR, RANGE, MDRANGE, TPX, TPTTRTVR, UNDEFINED };

enum Direction {IDIR, JDIR, KDIR};

#endif