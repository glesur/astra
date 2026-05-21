// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// These shortcuts are largely inspired from K-Athena
// https://gitlab.com/pgrete/kathena
// by P. Grete.

#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include <string>
#include <Kokkos_Core.hpp>
#include "field.hpp"
#ifdef DEBUG
#include "global.hpp"
#endif

// 1D default loop pattern
template <typename Function, typename Reducer>
inline void astra_reduce(const std::string & NAME,
                const int64_t & IB, const int64_t & IE,
                Function function,
                Reducer redFunction) {
    #ifdef DEBUG
    astra::pushRegion("astra_reduce("+NAME+")");
    #endif
    Kokkos::parallel_reduce(NAME,
      Kokkos::RangePolicy<>(IB,IE), function, redFunction);
    #ifdef DEBUG
    Kokkos::fence();
    astra::popRegion();
    #endif
}

// 2D default loop pattern
template <typename Function, typename Reducer>
inline void astra_reduce(const std::string & NAME,
                const int64_t & JB, const int64_t & JE,
                const int64_t & IB, const int64_t & IE,
                Function function,
                Reducer redFunction) {
    #ifdef DEBUG
    astra::pushRegion("astra_reduce("+NAME+")");
    #endif

    // We only implement MDRange reductions here since the other implementations are too
    // complicated to be implemented for any reduction operator on any class
    Kokkos::parallel_reduce(NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>
        ({JB,IB},{JE,IE}), function, redFunction);

    #ifdef DEBUG
    Kokkos::fence();
    astra::popRegion();
    #endif
}

// 3D default loop pattern
template <typename Function, typename Reducer>
inline void astra_reduce(const std::string & NAME,
                const int64_t & KB, const int64_t & KE,
                const int64_t & JB, const int64_t & JE,
                const int64_t & IB, const int64_t & IE,
                Function function,
                Reducer redFunction) {
    // We only implement MDRange reductions here since the other implementations are too
    // complicated to be implemented for any reduction operator on any class
    #ifdef DEBUG
    astra::pushRegion("astra_reduce("+NAME+")");
    #endif
    Kokkos::parallel_reduce(NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>
        ({KB,JB,IB},{KE,JE,IE}), function, redFunction);

    #ifdef DEBUG
    Kokkos::fence();
    astra::popRegion();
    #endif
}

// 4D default loop pattern
template <typename Function, typename Reducer>
inline void astra_reduce(const std::string & NAME,
                const int64_t & NB, const int64_t & NE,
                const int64_t & KB, const int64_t & KE,
                const int64_t & JB, const int64_t & JE,
                const int64_t & IB, const int64_t & IE,
                Function function,
                Reducer redFunction) {
    // We only implement MDRange reductions here since the other implementations are too
    // complicated to be implemented for any reduction operator on any class
    #ifdef DEBUG
    astra::pushRegion("astra_reduce("+NAME+")");
    #endif
    Kokkos::parallel_reduce(NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>
        ({NB,KB,JB,IB},{NE,KE,JE,IE}), function, redFunction);

    #ifdef DEBUG
    Kokkos::fence();
    astra::popRegion();
    #endif
}

// Generic function taking a Field class in parameter.
template <typename Function, typename Reducer, typename T>
  inline void astra_reduce(const std::string & NAME,
                       Field<T> fld,
                       Function function,
                       Reducer redFunction) {
    auto dims = fld.GetDimensions();

    if constexpr(T::rank == 1) {
      astra_reduce(NAME, 0, dims[0],function, redFunction);
    } else if constexpr(Field<T>::rank == 2) {
      astra_reduce(NAME, 0, dims[0],0,dims[1],function, redFunction);
    } else if constexpr(Field<T>::rank == 3) {
      astra_reduce(NAME, 0, dims[0],0,dims[1],0,dims[2],function, redFunction);
    } else if constexpr(Field<T>::rank == 4) {
      astra_reduce(NAME, 0, dims[0],0,dims[1],0,dims[2],0,dims[3],function, redFunction);
    }
  }


#endif // REDUCE_HPP_
