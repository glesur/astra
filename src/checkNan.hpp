// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <Kokkos_Core.hpp>

#include "arrays.hpp"
#include "field.hpp"
#include "reduce.hpp"

namespace astra {

template<typename ArrayType, typename = std::enable_if_t<Kokkos::is_view<ArrayType>::value>>
void CheckNan(ArrayType array) {
  astra::pushRegion("CheckNan(Array)");

  using value_type = typename ArrayType::value_type;
  constexpr bool is_complex = std::is_same<value_type, complex>::value;

  auto isNanValue = KOKKOS_LAMBDA(const value_type& v) -> bool {
    if constexpr (is_complex) {
      return std::isnan(v.real()) || std::isnan(v.imag());
    } else {
      return std::isnan(v);
    }
  };

  int nNan = 0;
  if constexpr(ArrayType::rank == 1) {
    astra_reduce("checkNan", 0, array.extent(0),
      KOKKOS_LAMBDA (const int64_t i, int& localCount) {
        if (isNanValue(array(i))) {
          localCount++;
        }
      }, Kokkos::Sum<int>(nNan));
    }
    else if constexpr(ArrayType::rank == 2) {
      astra_reduce("checkNan", 0, array.extent(0), 0, array.extent(1),
        KOKKOS_LAMBDA (const int64_t i, const int64_t j, int& localCount) {
          if (isNanValue(array(i,j))) {
            localCount++;
          }
        }, Kokkos::Sum<int>(nNan));
    } else if constexpr(ArrayType::rank == 3) {
      astra_reduce("checkNan", 0, array.extent(0), 0, array.extent(1), 0, array.extent(2),
        KOKKOS_LAMBDA (const int64_t i, const int64_t j, const int64_t k, int& localCount) {
          if (isNanValue(array(i,j,k))) {
            localCount++;
          }
        }, Kokkos::Sum<int>(nNan));
    } else if constexpr(ArrayType::rank == 4) {
      astra_reduce("checkNan", 0, array.extent(0), 0, array.extent(1), 0, array.extent(2), 0, array.extent(3),
        KOKKOS_LAMBDA (const int64_t i, const int64_t j, const int64_t k, const int64_t l, int& localCount) {
          if (isNanValue(array(i,j,k,l))) {
            localCount++;
          }
        }, Kokkos::Sum<int>(nNan));
    } else {
      throw std::runtime_error("checkNan: Unsupported array rank");
    }

  if (nNan > 0) {
    throw std::runtime_error("checkNan: NaN values found in array \""+array.label()+"\" (nan count: " + std::to_string(nNan) + ")");
  }
  astra::popRegion();
}

template<typename T>
void CheckNan(Field<T>& field) {
  astra::pushRegion("CheckNan(Field)");
  try{
     for(auto it : field) {
        CheckNan(it.second);  
      }
  } catch(const std::runtime_error& e) {
    throw std::runtime_error("NaN values found in Field \"" + field.GetName() + "\": " + e.what());
  }
  astra::popRegion();
}
}