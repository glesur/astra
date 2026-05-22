// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef ARRAYS_HPP_
#define ARRAYS_HPP_

#include <string>
#include <Kokkos_Core.hpp>
#include "astra.hpp"

namespace astra {
  // A helper function to unpack an array of dimensions into a variadic argument list for Array construction
  template <typename ViewType, typename T, std::size_t N>
  ViewType makeArray(const std::string& label, const std::array<T, N>& dims) {
    return std::apply(
        [&label](auto... args) { return ViewType(label, args...); },
        dims
    );
  }
}

template <typename T> using Array1D =
                            Kokkos::View<T*, Layout, Device>;
template <typename T> using Array2D =
                            Kokkos::View<T**, Layout, Device>;
template <typename T> using Array3D =
                            Kokkos::View<T***, Layout, Device>;
template <typename T> using Array4D =
                            Kokkos::View<T****, Layout, Device>;

template <typename T> using ArrayHost1D =
                            Kokkos::View<T*, Layout, Kokkos::HostSpace>;
template <typename T> using ArrayHost2D =
                            Kokkos::View<T**, Layout, Kokkos::HostSpace>;
template <typename T> using ArrayHost3D =
                            Kokkos::View<T***, Layout, Kokkos::HostSpace>;
template <typename T> using ArrayHost4D =
                            Kokkos::View<T****, Layout, Kokkos::HostSpace>;

// Atomic arrays
template <typename T> using ArrayAtomic1D =
                            Kokkos::View<T*, Layout, Device,
                                         Kokkos::MemoryTraits<Kokkos::Atomic>>;
template <typename T> using ArrayAtomic2D =
                            Kokkos::View<T**, Layout, Device,
                                         Kokkos::MemoryTraits<Kokkos::Atomic>>;
template <typename T> using ArrayAtomic3D =
                            Kokkos::View<T***, Layout, Device,
                                         Kokkos::MemoryTraits<Kokkos::Atomic>>;
/*
template <typename T> using ArrayHost1D = Kokkos::View<T*, Layout, Host>;
template <typename T> using ArrayHost2D = Kokkos::View<T**, Layout, Host>;
template <typename T> using ArrayHost3D = Kokkos::View<T***, Layout, Host>;
template <typename T> using ArrayHost4D = Kokkos::View<T****, Layout, Host>;
*/


#endif // ARRAYS_HPP_
