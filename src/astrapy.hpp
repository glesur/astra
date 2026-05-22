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
  void Output(Grid *, Field<Array3D<complex>> &, real, int );
  void InitFlow(Grid *, Field<ArrayHost3D<complex>> &);
  void ShowConfig();
  bool isActive{false};
  bool haveOutput{false};
  bool haveInitflow{false};
 private:
  template<typename... Ts>
  void CallScript(std::string, std::string, Ts...);
  static int ninstance;
  std::string scriptFilename;
  std::string outputFunctionName;
  std::string initflowFunctionName;
};


namespace pybind11 { namespace detail {
// Caster for IdefixArray4D<T>
template <typename T> struct type_caster<Array4D<T>> {
 public:
  PYBIND11_TYPE_CASTER(Array4D<T>, _("Array4D<T>"));

  // Conversion part 1 (Python -> C++)
  bool load(py::handle src, bool convert) {
    if ( !convert && !py::array_t<T>::check_(src) )
      return false;

    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
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
  static py::handle cast(const Array4D<T>& src,
                         py::return_value_policy policy,
                         py::handle parent) {
    py::none dummyDataOwner;
    py::array_t<real, py::array::c_style> a({src.extent(0),
                                             src.extent(1),
                                             src.extent(2),
                                             src.extent(3)},
                                             src.data(), dummyDataOwner);

    return a.release();
  }
};
// Caster for IdefixArray3D<T>
template <typename T> struct type_caster<Array3D<T>> {
 public:
  PYBIND11_TYPE_CASTER(Array3D<T>, _("Array3D<T>"));

  // Conversion part 1 (Python -> C++)
  bool load(py::handle src, bool convert) {
    astra::pushRegion("AstraPy::TypeCaster3D Python->C");
    if ( !convert && !py::array_t<T>::check_(src) )
      return false;

    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
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
  static py::handle cast(const Array3D<T>& src,
                         py::return_value_policy policy,
                         py::handle parent) {
    astra::pushRegion("AstraPy::TypeCaster3D C->Python");
    // Keep a local reference
    auto arr = src;
    py::none dummyDataOwner;
    py::array_t<real, py::array::c_style> a({src.extent(0),
                                             src.extent(1),
                                             src.extent(2)},
                                             src.data(),dummyDataOwner);
    astra::popRegion();
    return a.release();
  }
};
// Caster for IdefixArray2D<T>
template <typename T> struct type_caster<Array2D<T>> {
 public:
  PYBIND11_TYPE_CASTER(Array2D<T>, _("Array2D<T>"));

  // Conversion part 1 (Python -> C++)
  bool load(py::handle src, bool convert) {
    if ( !convert && !py::array_t<T>::check_(src) )
      return false;

    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
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
  static py::handle cast(const Array2D<T>& src,
                         py::return_value_policy policy,
                         py::handle parent) {
    py::none dummyOwner;
    py::array_t<real, py::array::c_style> a({src.extent(0),src.extent(1)},src.data(),dummyOwner);
    return a.release();
  }
};
// Caster for IdefixArray1D<T>
template <typename T> struct type_caster<Array1D<T>> {
 public:
  PYBIND11_TYPE_CASTER(Array1D<T>, _("Array1D<T>"));

  // Conversion part 1 (Python -> C++)
  bool load(py::handle src, bool convert) {
    if ( !convert && !py::array_t<T>::check_(src) )
      return false;

    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
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
  static py::handle cast(const Array1D<T>& src,
                         py::return_value_policy policy,
                         py::handle parent) {
    py::none dummyDataOwner;
    py::array_t<real, py::array::c_style> a(src.extent(0),src.data(),dummyDataOwner);
    return a.release();
  }
};
} // namespace detail
} // namespace pybind11

#endif // ASTRAPY_HPP_
