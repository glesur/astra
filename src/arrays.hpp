// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef ARRAYS_HPP_
#define ARRAYS_HPP_

#include <Kokkos_Core.hpp>
#include "astra.hpp"

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
