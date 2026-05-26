// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef ASTRAPY_HPP_
#define ASTRAPY_HPP_


#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <string>
#include <vector>
#include "astra.hpp"
#include "global.hpp"
#include "field.hpp"
#include "input.hpp"

class Grid;

namespace py = pybind11;

class DataBlock;
class DataBlockHost;

class AstraPy {
 public:
  explicit AstraPy(Input&);
  ~AstraPy();
  void Output(std::string, Grid *, Field<Array3D<complex>> &, real, int, bool realSpace = true);
  void InitFlow(std::string, Grid *, Field<ArrayHost3D<complex>> &, bool realSpace = true);
  void ShowConfig();
  bool isActive{false};
 private:
  template<typename... Ts>
  void CallScript(std::string, std::string, Ts...);
  static int ninstance;
  std::string scriptFilename;
};


namespace pybind11 { namespace detail {

  // Type trait: strip Kokkos::complex → std::complex ──────────────────────
template <typename T>
struct numpy_type {
    using type = T;  // pass-through for float, double, int, etc.
};

template <typename T>
struct numpy_type<Kokkos::complex<T>> {
    using type = std::complex<T>;  // Kokkos::complex<double> → std::complex<double>
};

// Convenience alias
template <typename T>
using numpy_type_t = typename numpy_type<T>::type;

// Caster for IdefixArray4D<T>
template <typename T> struct type_caster<ArrayHost4D<T>> {
 public:
  PYBIND11_TYPE_CASTER(ArrayHost4D<T>, _("ArrayHost4D<T>"));

  // Conversion part 1 (Python -> C++)
  bool load(py::handle src, bool convert) {
    if ( !convert && !py::array_t<T>::check_(src) )
      return false;

    using NumpyT = numpy_type_t<T>;
    auto array = py::array_t<NumpyT, py::array::c_style | py::array::forcecast>::ensure(src);
    if ( !array )
      return false;

    auto dims = array.ndim();
    if ( dims != 4  )
      return false;

    auto buf = array.request();



    value = Kokkos::View<T****,
                        Kokkos::LayoutRight,
                        Kokkos::HostSpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>> (reinterpret_cast<T*>(buf.ptr),
                                                                  array.shape()[0],
                                                                  array.shape()[1],
                                                                  array.shape()[2],
                                                                  array.shape()[3]);


    return true;
  }

  //Conversion part 2 (C++ -> Python)
  static py::handle cast(const ArrayHost4D<T>& src,
                         py::return_value_policy policy,
                         py::handle parent) {
    py::none dummyDataOwner;
    using NumpyT = numpy_type_t<T>;
    py::array_t<NumpyT, py::array::c_style> a({src.extent(0),
                                                 src.extent(1),
                                                 src.extent(2),
                                                 src.extent(3)},
                                                 reinterpret_cast<const NumpyT*>(src.data()), dummyDataOwner);

    return a.release();
  }
};
// Caster for IdefixArray3D<T>
template <typename T> struct type_caster<ArrayHost3D<T>> {
 public:
  PYBIND11_TYPE_CASTER(ArrayHost3D<T>, _("ArrayHost3D<T>"));

  // Conversion part 1 (Python -> C++)
  bool load(py::handle src, bool convert) {
    astra::pushRegion("AstraPy::TypeCaster3D Python->C");
    if ( !convert && !py::array_t<T>::check_(src) )
      return false;

    using NumpyT = numpy_type_t<T>;
    auto array = py::array_t<NumpyT, py::array::c_style | py::array::forcecast>::ensure(src);
    if ( !array )
      return false;

    auto dims = array.ndim();
    if ( dims != 3  )
      return false;

    auto buf = array.request();

    value = Kokkos::View<T***,
                        Kokkos::LayoutRight,
                        Kokkos::HostSpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>> (reinterpret_cast<T*>(buf.ptr),
                                                                  array.shape()[0],
                                                                  array.shape()[1],
                                                                  array.shape()[2]);

    astra::popRegion();
    return true;
  }

  //Conversion part 2 (C++ -> Python)
  static py::handle cast(const ArrayHost3D<T>& src,
                         py::return_value_policy policy,
                         py::handle parent) {
    astra::pushRegion("AstraPy::TypeCaster3D C->Python");
    // Keep a local reference
    auto arr = src;
    py::none dummyDataOwner;
    using NumpyT = numpy_type_t<T>;
    py::array_t<NumpyT, py::array::c_style> a({src.extent(0),
                                                 src.extent(1),
                                                 src.extent(2)},
                                                 reinterpret_cast<const NumpyT*>(src.data()), dummyDataOwner);
    astra::popRegion();
    return a.release();
  }
};
// Caster for IdefixArray2D<T>
template <typename T> struct type_caster<ArrayHost2D<T>> {
 public:
  PYBIND11_TYPE_CASTER(ArrayHost2D<T>, _("ArrayHost2D<T>"));

  // Conversion part 1 (Python -> C++)
  bool load(py::handle src, bool convert) {
    if ( !convert && !py::array_t<T>::check_(src) )
      return false;

    using NumpyT = numpy_type_t<T>;
    auto array = py::array_t<NumpyT, py::array::c_style | py::array::forcecast>::ensure(src);
    if ( !array )
      return false;

    auto dims = array.ndim();
    if ( dims != 2  )
      return false;

    auto buf = array.request();

    value = Kokkos::View<T**,
                        Kokkos::LayoutRight,
                        Kokkos::HostSpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>> (reinterpret_cast<T*>(buf.ptr),
                                                                  array.shape()[0],
                                                                  array.shape()[1]);

    astra::popRegion();
    return true;
  }

  //Conversion part 2 (C++ -> Python)
  static py::handle cast(const ArrayHost2D<T>& src,
                         py::return_value_policy policy,
                         py::handle parent) {
    using NumpyT = numpy_type_t<T>;
    py::none dummyOwner;
    py::array_t<NumpyT, py::array::c_style> a({src.extent(0),src.extent(1)},reinterpret_cast<const NumpyT*>(src.data()),dummyOwner);
    return a.release();
  }
};
// Caster for IdefixArray1D<T>
template <typename T> struct type_caster<ArrayHost1D<T>> {
 public:
  PYBIND11_TYPE_CASTER(ArrayHost1D<T>, _("ArrayHost1D<T>"));

  // Conversion part 1 (Python -> C++)
  bool load(py::handle src, bool convert) {
    if ( !convert && !py::array_t<T>::check_(src) )
      return false;

    using NumpyT = numpy_type_t<T>;
    auto array = py::array_t<NumpyT, py::array::c_style | py::array::forcecast>::ensure(src);
    if ( !array )
      return false;

    auto dims = array.ndim();
    if ( dims != 1  )
      return false;

    auto buf = array.request();

    value = Kokkos::View<T*,
                        Kokkos::LayoutRight,
                        Kokkos::HostSpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>> (reinterpret_cast<T*>(buf.ptr),
                                                                  array.shape()[0]);

    astra::popRegion();
    return true;
  }

  //Conversion part 2 (C++ -> Python)
  static py::handle cast(const ArrayHost1D<T>& src,
                         py::return_value_policy policy,
                         py::handle parent) {
    using NumpyT = numpy_type_t<T>;
    py::none dummyDataOwner;
    py::array_t<NumpyT, py::array::c_style> a(src.extent(0),reinterpret_cast<const NumpyT*>(src.data()),dummyDataOwner);
    return a.release();
  }
};
} // namespace detail
} // namespace pybind11

#endif // ASTRAPY_HPP_
