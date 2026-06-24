// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "astrapy.hpp"
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h> // for numpy arrays
#include <pybind11/stl.h>   // For STL vectors and containers
#include <csignal>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include "astra.hpp"
#include "arrays.hpp"
#include "input.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "gridhost.hpp"
#include "fft.hpp"


namespace py = pybind11;

int AstraPy::ninstance = 0;


namespace AstraPyTools {
// Functions provided by Idefix in AstraPy for user convenience
py::array_t<real, py::array::c_style> GatherAstraArray(ArrayHost3D<real> in,
                                                        GridHost gridHost,
                                                        bool keepBoundaries = true,
                                                        bool broadcast = true) {
  astra::pushRegion("AstraPyTools::GatherAstraArray");
  // To be done
  throw std::runtime_error("AstraPyTools::GatherAstraArray is not implemented yet");
  astra::popRegion();
}
}// namespace AstraPyTools

/************************************
 * DataBlockHost Python binding
 * **********************************/

PYBIND11_EMBEDDED_MODULE(astrapy, m) {
  m.doc() = "AstraPy: Python interface for ASTRA spectral code";
  py::class_<GridHost>(m, "GridHost")
    .def(py::init<>())
    .def_readwrite("x", &GridHost::x)
    .def_readwrite("x_glob", &GridHost::x_glob)
    .def_readwrite("kx", &GridHost::kx)
    .def_readwrite("kx_glob", &GridHost::kx_glob)
    .def_readwrite("dx", &GridHost::dx)
    .def_readwrite("xbeg", &GridHost::xbeg_glob)
    .def_readwrite("xend", &GridHost::xend_glob)
    .def_readwrite("xbeg", &GridHost::xbeg)
    .def_readwrite("xend", &GridHost::xend)
    .def_readwrite("npr_glob", &GridHost::npr_glob)
    .def_readwrite("npr", &GridHost::npr)
    .def_readwrite("npf_glob", &GridHost::npf_glob)
    .def_readwrite("npf", &GridHost::npf);

      m.attr("IDIR") = 0;
      m.attr("JDIR") = 1;
      m.attr("KDIR") = 2;

      m.attr("prank") = astra::prank;
      m.attr("psize") = astra::psize;

/*
    m.def("GatherAstraArray",&AstraPyTools::GatherAstraArray,
                               py::arg("in"),
                               py::arg("data"),
                               py::arg("keepBoundaries") = true,
                               py::arg("broadcast") = true,
                               "Gather arrays from MPI domain decomposition");*/
}


template<typename... Ts>
void AstraPy::CallScript(std::string scriptName, std::string funcName, Ts... args) {
  astra::pushRegion("AstraPy::CallScript");
  try {
    py::module_ script = py::module_::import(scriptName.c_str());
    py::object result = script.attr(funcName.c_str())(&args...);
  } catch(std::exception &e) {
    std::stringstream message;
    message << "An exception occured while calling the Python interpreter" << std::endl
                << "in file \"" << scriptName << ".py\" function \"" << funcName << "\":"
                << std::endl
                << e.what() << std::endl;
    throw std::runtime_error(message.str());
  }
  astra::popRegion();
}


AstraPy::AstraPy(Input &input) {
  // Check that the input has a [python] block
  if(input.CheckBlock("Python")) {
    this->isActive = true;
    ninstance++;
    // Check whether we need to start an interpreter
    if(ninstance==1) {
      astra::cout << "AstraPy: start Python interpreter." << std::endl;
      py::initialize_interpreter();
      std::signal(SIGINT, SIG_DFL);  // restore "terminate on Ctrl-C" signal handler

      py::exec("import sys; print(f'AstraPy: Python Version: {sys.version}')");
      py::exec("print(f'AstraPy: Executable Path: {sys.executable}')");
      py::exec("print(f'AstraPy: Sys Path: {sys.path}')");
    }
    this->scriptFilename = input.Get<std::string>("Python","script",0);
    if(scriptFilename.substr(scriptFilename.length() - 3, 3).compare(".py")==0) {
      throw std::runtime_error("You should not include the python script .py extension in your input file");
    }
  }
}

void AstraPy::Output(std::string outputFunctionName, Grid *grid, Field<Array3D<complex>> &field, real time, int n, bool realSpace) {
  astra::pushRegion("AstraPy::Output");
  astra::cout << "Astrapy: Python output n " << n << "..." << std::flush;
  if(!this->isActive) {
    throw std::runtime_error("Python Outputs requires the [python] block to be defined in your input file.");
  }

  GridHost gridHost(*grid);

  if(realSpace) {
    // Fourier-transform the field to real space on host, and store in a map
    std::map<std::string, ArrayHost3D<real>> mapFieldReal;
    for(auto const& [name, view] : field) {
      // Fourier transform each view
      Array3D<real> deviceReal = astra::makeArray<Array3D<real>>("deviceReal", grid->npr);
      grid->fft->C2R(view, deviceReal);
      ArrayHost3D<real> hostReal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), deviceReal);
      // Store in the map
      mapFieldReal[name] = hostReal;
    }

    // todo(lesurg): remap the flow
    // Call the Python function, passing the grid and field data
    this->CallScript(this->scriptFilename,outputFunctionName, gridHost, mapFieldReal, time, n);
  } else {
    // copy the complex field in the map
    std::map<std::string, ArrayHost3D<complex>> mapFieldComplex;
    for(auto const& [name, view] : field) {
      // Store in the map
      mapFieldComplex[name] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),view);
    }

    this->CallScript(this->scriptFilename,outputFunctionName, gridHost, mapFieldComplex, time, n);
  }
  astra::cout << "done." << std::endl;
  astra::popRegion();
}

void AstraPy::InitFlow(std::string initflowFunctionName, Grid *grid, Field<ArrayHost3D<complex>> &field, bool realSpace) {
  astra::pushRegion("AstraPy::InitFlow");
  astra::cout << "Astrapy: calling python initflow function..." << std::flush;
  if(!this->isActive) {
    throw std::runtime_error("Python Initflow requires the [python] block to be defined in your input file.");
  }

  GridHost gridHost(*grid);

  if(realSpace) {
    // Fourier-transform the field to real space on host, and store in a map
    std::map<std::string, ArrayHost3D<real>> mapFieldReal;
    for(auto const& [name, view] : field) {
      // Fourier transform each view
      ArrayHost3D<real> hostReal = astra::makeArray<ArrayHost3D<real>>("hostView", grid->npr);
      grid->fft->C2R_Host(view, hostReal);
      // Store in the map
      mapFieldReal[name] = hostReal;
    }

    // Call the Python function, passing the grid and field data
    this->CallScript(this->scriptFilename,initflowFunctionName,gridHost, mapFieldReal);

    //Transform back
    for(auto const& [name, realView] : mapFieldReal) {
      grid->fft->R2C_Host(realView, field[name]);
    }
  } else {
    // Fourier-transform the field to real space on host, and store in a map
    std::map<std::string, ArrayHost3D<complex>> mapFieldComplex;
    for(auto const& [name, view] : field) {
      // Store in the map
      mapFieldComplex[name] = view;
    }

    // Call the Python function, passing the grid and field data
    this->CallScript(this->scriptFilename,initflowFunctionName,gridHost, mapFieldComplex);
    // nothing to be done to transform back since the Python function directly modifies the complex field in Fourier space
  }
  astra::cout << "done." << std::endl;
  astra::popRegion();
}

void AstraPy::ShowConfig() {
  if(isActive == false) {
    astra::cout << "AstraPy: DISABLED." << std::endl;
  } else {
    astra::cout << "AstraPy: ENABLED." << std::endl;
  }
}

AstraPy::~AstraPy() {
  if(isActive) {
    if(ninstance == 1) {
      py::finalize_interpreter();
      astra::cout << "AstraPy: shutdown Python interpreter." << std::endl;
    }
    ninstance--;
    isActive = false;
  }
}


/*
py::array_t<real> AstraPy::toNumpyArray(const ArrayHost3D<real>& in) {
  py::array_t<real, py::array::c_style> array({in.extent(0),in.extent(1),in.extent(2)},in.data());
  return array;
}

py::array_t<real> AstraPy::toNumpyArray(const ArrayHost4D<real>& in) {
  py::array_t<real, py::array::c_style> array({in.extent(0),in.extent(1),in.extent(2),in.extent(3)},in.data());
  return array;
}
*/
