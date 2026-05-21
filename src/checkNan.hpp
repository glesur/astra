// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef CHECKNAN_HPP_
#define CHECKNAN_HPP_

#include <Kokkos_Core.hpp>

#include "arrays.hpp"
#include "field.hpp"
#include "reduce.hpp"

namespace astra {

KOKKOS_INLINE_FUNCTION bool isNanValue(double v) {
  return std::isnan(v);
}
// Specialization for complex numbers
KOKKOS_INLINE_FUNCTION bool isNanValue(const complex& v) {
  return std::isnan(v.real()) || std::isnan(v.imag());
}

template<typename ArrayType>
int CheckNan1D(ArrayType array) {
  int nNan = 0;
  astra_reduce("checkNan", 0, array.extent(0),
    KOKKOS_LAMBDA (const int64_t i, int& localCount) {
      if (isNanValue(array(i))) localCount++;
    }, Kokkos::Sum<int>(nNan));
  return nNan;
}

template<typename ArrayType>
int CheckNan2D(ArrayType array) {
  int nNan = 0;
  astra_reduce("checkNan", 0, array.extent(0),
               0, array.extent(1),
    KOKKOS_LAMBDA (const int64_t i, const int64_t j, int& localCount) {
      if (isNanValue(array(i,j))) localCount++;
    }, Kokkos::Sum<int>(nNan));
  return nNan;
}

template<typename ArrayType>
int CheckNan3D(ArrayType array) {
  int nNan = 0;
  astra_reduce("checkNan", 0, array.extent(0),
               0, array.extent(1),
               0, array.extent(2),
    KOKKOS_LAMBDA (const int64_t i, const int64_t j, const int64_t k, int& localCount) {
      if (isNanValue(array(i,j,k))) localCount++;
    }, Kokkos::Sum<int>(nNan));

  return nNan;
}

template<typename ArrayType, typename = std::enable_if_t<Kokkos::is_view<ArrayType>::value>>
void CheckNan(ArrayType array) {
  astra::pushRegion("CheckNan(Array)");
  int nNan = 0;
  if constexpr(ArrayType::rank == 1)      nNan = CheckNan1D(array);
  else if constexpr(ArrayType::rank == 2) nNan = CheckNan2D(array);
  else if constexpr(ArrayType::rank == 3) nNan = CheckNan3D(array);
  else static_assert(ArrayType::rank > 3, "CheckNan: unsupported rank");

  if (nNan > 0) {
    throw std::runtime_error("checkNan: NaN values found in array \""+array.label()+"\" (nan count: " + std::to_string(nNan) + ")");
  }
  astra::popRegion();
}

template<typename T>
void CheckNan(Field<T>& field) {
  astra::pushRegion("CheckNan(Field)");
  try {
     for(auto it : field) {
        CheckNan(it.second);
      }
  } catch(const std::runtime_error& e) {
    throw std::runtime_error("NaN values found in Field \"" + field.GetName() + "\": " + e.what());
  }
  astra::popRegion();
}
} // namespace astra

#endif // CHECKNAN_HPP_
