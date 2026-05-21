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

#ifndef LOOP_HPP_
#define LOOP_HPP_

#include <string>
#include "astra.hpp"
#include "field.hpp"

#define KOKKOS_VECTOR_LENGTH  8


#ifdef INNER_TTR_LOOP
  #define TPINNERLOOP Kokkos::TeamThreadRange
#elif defined INNER_TVR_LOOP
  #define TPINNERLOOP Kokkos::ThreadVectorRange
#else
  #define TPINNERLOOP Kokkos::TeamThreadRange
#endif

typedef Kokkos::TeamPolicy<>               team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;



// Check if the user requested a specific loop unrolling strategy
#if defined(LOOP_PATTERN_SIMD)
  constexpr LoopPattern defaultLoop = LoopPattern::SIMDFOR;
#elif defined(LOOP_PATTERN_1DRANGE)
  constexpr LoopPattern defaultLoop = LoopPattern::RANGE;
#elif defined(LOOP_PATTERN_MDRANGE)
  constexpr LoopPattern defaultLoop = LoopPattern::MDRANGE;
#elif defined(LOOP_PATTERN_TPX)
  constexpr LoopPattern defaultLoop = LoopPattern::TPX;
#elif defined(LOOP_PATTERN_TPTTRTVR)
  constexpr LoopPattern defaultLoop = LoopPattern::TPTTRTVR;
#else // no loop strategy has been defined
  // Default loops
  #if defined(KOKKOS_ENABLE_OPENMP)
    constexpr LoopPattern defaultLoop = LoopPattern::TPTTRTVR;
  #elif defined(KOKKOS_ENABLE_CUDA)
    constexpr LoopPattern defaultLoop = LoopPattern::RANGE;
  #elif defined(KOKKOS_ENABLE_HIP)
    constexpr LoopPattern defaultLoop = LoopPattern::RANGE;
  #elif defined(KOKKOS_ENABLE_SERIAL)
    constexpr LoopPattern defaultLoop = LoopPattern::SIMDFOR;
  #else
    #warning "Unknown target architeture: default to MDrange"
    constexpr LoopPattern defaultLoop = LoopPattern::MDRANGE;
  #endif
#endif



// 1D loop
template <typename Function>
inline void astra_for(const std::string & NAME,
                       const int64_t & IB, const int64_t & IE,
                       Function function) {
  const int64_t NI = IE - IB;
  Kokkos::parallel_for(NAME, NI,
    KOKKOS_LAMBDA (const int& IDX) {
      int64_t i = IDX;
      i += IB;
      function(i);
  });
}


// 2D loop
template <typename Function>
inline void astra_for(const std::string & NAME,
                       const int64_t & JB, const int64_t & JE,
                       const int64_t & IB, const int64_t & IE,
                       Function function) {
  // Kokkos 1D Range
  if constexpr(defaultLoop == LoopPattern::RANGE) {
    const int64_t NJ = JE - JB;
    const int64_t NI = IE - IB;
    const int64_t NJNI = NJ * NI;
    Kokkos::parallel_for(NAME, NJNI,
      KOKKOS_LAMBDA (const int& IDX) {
        int64_t j = IDX  / NI;
        int64_t i = IDX - j*NI;
        j += JB;
        i += IB;
        function(j,i);
    });

    // MDRange loops
  } else if constexpr(defaultLoop == LoopPattern::MDRANGE) {
    Kokkos::parallel_for(NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>
        ({JB,IB},{JE,IE}), function);

    // TeamPolicies with single inner loops
  } else if constexpr(defaultLoop == LoopPattern::TPX || defaultLoop == LoopPattern::TPTTRTVR ) {
    const int64_t NJ = JE - JB;
    Kokkos::parallel_for(NAME, team_policy (NJ, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        const int64_t j = team_member.league_rank() + JB;
        Kokkos::parallel_for(TPINNERLOOP<>(team_member,IB,IE),
                             [&] (const int64_t i) {
                               function(j,i);
                            });
    });

    // SIMD FOR loops
  } else if constexpr(defaultLoop == LoopPattern::SIMDFOR) {
    for (auto j = JB; j < JE; j++)
#pragma omp simd
      for (auto i = IB; i < IE; i++)
        function(j,i);
  } else {
    throw std::runtime_error("Unknown/undefined LoopPattern used.");
  }
}


// 3D loop
template <typename Function>
inline void astra_for(const std::string & NAME,
                       const int64_t & KB, const int64_t & KE,
                       const int64_t & JB, const int64_t & JE,
                       const int64_t & IB, const int64_t & IE,
                       Function function) {
  // Kokkos 1D Range
  if constexpr(defaultLoop == LoopPattern::RANGE) {
    const int64_t NK = KE - KB;
    const int64_t NJ = JE - JB;
    const int64_t NI = IE - IB;
    const int64_t NKNJNI = NK*NJ*NI;
    const int64_t NJNI = NJ * NI;
    Kokkos::parallel_for(NAME,NKNJNI,
      KOKKOS_LAMBDA (const int& IDX) {
        int64_t k = IDX / NJNI;
        int64_t j = (IDX - k*NJNI) / NI;
        int64_t i = IDX - k*NJNI - j*NI;
        k += KB;
        j += JB;
        i += IB;
        function(k,j,i);
    });

  // MDRange loops
  } else if constexpr(defaultLoop == LoopPattern::MDRANGE) {
    Kokkos::parallel_for(NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>
        ({KB,JB,IB},{KE,JE,IE}), function);

  // TeamPolicy with single inner loops
  } else if constexpr(defaultLoop == LoopPattern::TPX) {
    const int64_t NK = KE - KB;
    const int64_t NJ = JE - JB;
    const int64_t NKNJ = NK * NJ;
    Kokkos::parallel_for(NAME,
      team_policy (NKNJ, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        const int64_t k = team_member.league_rank() / NJ + KB;
        const int64_t j = team_member.league_rank() % NJ + JB;
        Kokkos::parallel_for(TPINNERLOOP<>(team_member,IB,IE),
          [&] (const int64_t i) {
            function(k,j,i);
        });
      });

  // TeamPolicy with nested TeamThreadRange and ThreadVectorRange
  } else if constexpr(defaultLoop == LoopPattern::TPTTRTVR) {
    const int64_t NK = KE - KB;
    Kokkos::parallel_for(NAME,
      team_policy (NK, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        const int64_t k = team_member.league_rank() + KB;
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange<>(team_member,JB,JE),
          [&] (const int64_t j) {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange<>(team_member,IB,IE),
              [&] (const int64_t i) {
                function(k,j,i);
              });
          });
      });

  // SIMD FOR loops
  } else if constexpr(defaultLoop == LoopPattern::SIMDFOR) {
    for (auto k = KB; k < KE; k++)
      for (auto j = JB; j < JE; j++)
#pragma omp simd
        for (auto i = IB; i < IE; i++)
          function(k,j,i);
  } else {
    throw std::runtime_error("Unknown/undefined LoopPattern used.");
  }
}

// 4D loop
template <typename Function>
inline void astra_for(const std::string & NAME,
                       const int64_t NB, const int64_t NE,
                       const int64_t KB, const int64_t KE,
                       const int64_t JB, const int64_t JE,
                       const int64_t IB, const int64_t IE,
                       Function function) {
  // Kokkos 1D Range
  if constexpr(defaultLoop == LoopPattern::RANGE) {
    const int64_t NN = (NE) - (NB);
    const int64_t NK = (KE) - (KB);
    const int64_t NJ = (JE) - (JB);
    const int64_t NI = (IE) - (IB);
    const int64_t NNNKNJNI = NN*NK*NJ*NI;
    const int64_t NKNJNI = NK*NJ*NI;
    const int64_t NJNI = NJ * NI;
    Kokkos::parallel_for(NAME,NNNKNJNI,
      KOKKOS_LAMBDA (const int& IDX) {
        int64_t n = IDX / NKNJNI;
        int64_t k = (IDX - n*NKNJNI) / NJNI;
        int64_t j = (IDX - n*NKNJNI - k*NJNI) / NI;
        int64_t i = IDX - n*NKNJNI - k*NJNI - j*NI;
        n += (NB);
        k += (KB);
        j += (JB);
        i += (IB);
        function(n,k,j,i);
    });

  // MDRange loops
  } else if constexpr(defaultLoop == LoopPattern::MDRANGE) {
    Kokkos::parallel_for(NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<4,Kokkos::Iterate::Left, Kokkos::Iterate::Left>>
        ({NB,KB,JB,IB},{NE,KE,JE,IE}), function);

  // TeamPolicy loops
  } else if constexpr(defaultLoop == LoopPattern::TPX) {
    const int64_t NN = NE - NB;
    const int64_t NK = KE - KB;
    const int64_t NJ = JE - JB;
    const int64_t NKNJ = NK * NJ;
    const int64_t NNNKNJ = NN * NK * NJ;
    Kokkos::parallel_for(NAME,
      team_policy (NNNKNJ, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        int64_t n = team_member.league_rank() / NKNJ;
        int64_t k = (team_member.league_rank() - n*NKNJ) / NJ;
        int64_t j = team_member.league_rank() - n*NKNJ - k*NJ + JB;
        n += NB;
        k += KB;
        Kokkos::parallel_for(
          TPINNERLOOP<>(team_member,IB,IE),
          [&] (const int64_t i) {
            function(n,k,j,i);
          });
      });

  // TeamPolicy with nested TeamThreadRange and ThreadVectorRange
  } else if constexpr(defaultLoop == LoopPattern::TPTTRTVR) {
    const int64_t NN = NE - NB;
    const int64_t NK = KE - KB;
    const int64_t NNNK = NN * NK;
    Kokkos::parallel_for(NAME,
      team_policy (NNNK, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        int64_t n = team_member.league_rank() / NK + NB;
        int64_t k = team_member.league_rank() % NK + KB;
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange<>(team_member,JB,JE),
          [&] (const int64_t j) {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange<>(team_member,IB,IE),
              [&] (const int64_t i) {
                function(n,k,j,i);
              });
          });
      });

  // SIMD FOR loops
  } else if constexpr(defaultLoop == LoopPattern::SIMDFOR) {
    for (auto n = NB; n < NE; n++)
      for (auto k = KB; k < KE; k++)
        for (auto j = JB; j < JE; j++)
#pragma omp simd
          for (auto i = IB; i < IE; i++)
            function(n,k,j,i);
  } else {
    throw std::runtime_error("Unknown/undefined LoopPattern used.");
  }
}

// Generic function taking a Field class in parameter.
template <typename Function, typename T>
  inline void astra_for(const std::string & NAME,
                       Field<T> fld,
                       Function function) {
    auto dims = fld.GetDimensions();

    if constexpr(T::rank == 1) {
      astra_for(NAME, 0, dims[0],function);
    } else if constexpr(Field<T>::rank == 2) {
      astra_for(NAME, 0, dims[0],0,dims[1],function);
    } else if constexpr(Field<T>::rank == 3) {
      astra_for(NAME, 0, dims[0],0,dims[1],0,dims[2],function);
    } else if constexpr(Field<T>::rank == 4) {
      astra_for(NAME, 0, dims[0],0,dims[1],0,dims[2],0,dims[3],function);
    }
  }

#endif // LOOP_HPP_
