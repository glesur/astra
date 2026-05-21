// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef ASTRA_HPP_
#define ASTRA_HPP_
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

using Device = Kokkos::DefaultExecutionSpace;
using Layout = Kokkos::LayoutRight;
using real = double;
using complex = Kokkos::complex<real>;

/// Type of loops we admit in idefix (see loop.hpp for details)
enum class LoopPattern { SIMDFOR, RANGE, MDRANGE, TPX, TPTTRTVR, UNDEFINED };

enum Direction {IDIR, JDIR, KDIR};

#endif // ASTRA_HPP_
