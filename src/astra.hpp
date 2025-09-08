#ifndef ASTRA_HPP_
#define ASTRA_HPP_
#include <Kokkos_Core.hpp>

using Device = Kokkos::DefaultExecutionSpace;
using Layout = Kokkos::LayoutLeft;


/// Type of loops we admit in idefix (see loop.hpp for details)
enum class LoopPattern { SIMDFOR, RANGE, MDRANGE, TPX, TPTTRTVR, UNDEFINED };

#include "loop.hpp"
#include "arrays.hpp"
#endif