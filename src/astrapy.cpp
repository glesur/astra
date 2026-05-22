// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "astrapy.hpp"
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h> // for numpy arrays
#include <pybind11/stl.h>   // For STL vectors and containers
#include <map>
#include <string>
#include <vector>
#include "astra.hpp"
#include "arrays.hpp"
#include "input.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "gridhost.hpp"


namespace py = pybind11;

int AstraPy::ninstance = 0;


namespace AstraPyTools {
// Functions provided by Idefix in AstraPy for user convenience
py::array_t<real, py::array::c_style> GatherAstraArray(ArrayHost3D<real> in,
                                                        DataBlockHost dataHost,
                                                        bool keepBoundaries = true,
                                                        bool broadcast = true) {
  astra::pushRegion("AstraPyTools::GatherAstraArray");
  // To be done
  throw std::runtime_error("AstraPyTools::GatherAstraArray is not implemented yet");
  astra::popRegion();
  return pyOut;
}
}// namespace AstraPyTools

/************************************
 * DataBlockHost Python binding
 * **********************************/

PYBIND11_EMBEDDED_MODULE(astrapy, m) {
  py::class_<GridHost>(m, "GridHost")
    .def(py::init<>())
    .def_readwrite("x", &GridHost::x)
    .def_readwrite("x_glob", &GridHost::x_glob)
    .def_readwrite("kx", &GridHost::kx)
    .def_readwrite("kx_glob", &GridHost::kx_glob);
    .def_readwrite("dx", &GridHost::dx)
    .def_readwrite("xbeg", &GridHost::xbeg_glob)
    .def_readwrite("xend", &GridHost::xend_glob)
    .def_readwrite("xbeg", &GridHost::xbeg)
    .def_readwrite("xend", &GridHost::xend)
    .def_readwrite("npr_glob", &GridHost::npr_glob)
    .def_readwrite("npr", &GridHost::npr)
    .def_readwrite("npf_glob", &GridHost::npf_glob)
    .def_readwrite("npf", &GridHost::npf);



    m.attr("IDIR") = IDIR;
    m.attr("JDIR") = JDIR;
    m.attr("KDIR") = KDIR;

    m.attr("prank") = astra::prank;
    m.attr("psize") = astra::psize;

    m.def("GatherAstraArray",&AstraPyTools::GatherAstraArray,
                               py::arg("in"),
                               py::arg("data"),
                               py::arg("keepBoundaries") = true,
                               py::arg("broadcast") = true,
                               "Gather arrays from MPI domain decomposition");
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
    IDEFIX_ERROR(message);
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
      py::exec("import sys; print(f'AstraPy: Python Version: {sys.version}')");
      py::exec("print(f'AstraPy: Executable Path: {sys.executable}')");
      py::exec("print(f'AstraPy: Sys Path: {sys.path}')");
    }
    this->scriptFilename = input.Get<std::string>("Python","script",0);
    if(scriptFilename.substr(scriptFilename.length() - 3, 3).compare(".py")==0) {
      IDEFIX_ERROR("You should not include the python script .py extension in your input file");
    }
    if(input.CheckEntry("Python","output_function")>0) {
      this->haveOutput = true;
      this->outputFunctionName = input.Get<std::string>("Python","output_function",0);
    }
    if(input.CheckEntry("Python","initflow_function")>0) {
      this->haveInitflow = true;
      this->initflowFunctionName = input.Get<std::string>("Python","initflow_function",0);
    }
  }
}

void AstraPy::Output(Grid *grid, Field<Array3D<complex>> &field, real time, int n) {
  astra::pushRegion("AstraPy::Output");
  if(!this->isActive) {
    IDEFIX_ERROR("Python Outputs requires the [python] block to be defined in your input file.");
  }
  if(!this->haveOutput) {
    IDEFIX_ERROR("No python output function has been defined "
                  "in your input file [python]:output_function");
  }
  GridHost gridHost(*grid);

  this->CallScript(this->scriptFilename,this->outputFunctionName, gridHost, time, n);
  astra::popRegion();
}

void AstraPy::InitFlow(Grid *grid, Field<ArrayHost3D<complex>> &field) {
  astra::pushRegion("AstraPy::InitFlow");
  if(!this->isActive) {
    IDEFIX_ERROR("Python Initflow requires the [python] block to be defined in your input file.");
  }
  if(!this->haveInitflow) {
    IDEFIX_ERROR("No python initflow function has been defined "
                  "in your input file [python]:initflow_function");
  }

  GridHost gridHost(*grid);

  // Make a map-host copy of the field data to pass to Python
  std::map<std::string, ArrayHost3D<complex>> fieldHost;

  for(auto const& [name, view] : field) {
    fieldHost[name] = view;
  }

  // Call the Python function, passing the grid and field data
  this->CallScript(this->scriptFilename,this->initflowFunctionName,gridHost, fieldHost);

  // Copy back the field data from host to device


  astra::popRegion();
}

void AstraPy::ShowConfig() {
  if(isActive == false) {
    astra::cout << "AstraPy: DISABLED." << std::endl;
  } else {
    astra::cout << "AstraPy: ENABLED." << std::endl;
    if(haveOutput) {
      astra::cout << "AstraPy: output function ENABLED." << std::endl;
    } else {
      astra::cout << "AstraPy: output function DISABLED." << std::endl;
    }
    if(haveInitflow) {
      astra::cout << "AstraPy: initflow function ENABLED." << std::endl;
    } else {
      astra::cout << "AstraPy: initflow function DISABLED." << std::endl;
    }
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
