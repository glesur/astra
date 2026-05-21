// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************


#ifndef OUTPUT_SLICE_HPP_
#define OUTPUT_SLICE_HPP_

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <memory>
#include <map>
#include <string>
#include "arrays.hpp"
#include "field.hpp"
#include "input.hpp"
#include "subgrid.hpp"

class SubGrid;
class Vtk;
class LinearShear;
class Output;

class Slice {
 public:
  Slice();
  Slice(Input &, Grid *, int nSlice);
  void CheckForWrite(Field<Array3D<complex>> state, real time, Output *output);
  void WriteSlice(Array3D<real> slice, Vtk *vtk, real time, std::string name);

 private:
  // vtk output periods
  real slicePeriod = 0.0;
  real sliceLast = 0.0;
  std::string outputVtkDirectory{"./"};
  int nvtk = 0;

  bool containsX0;
  real x0;
  SliceType type;
  int direction;
  std::unique_ptr<SubGrid> subGrid;
  int nSlice;   // slice number, used for output file naming
  Input *input;

  bool haveShear{false};
  std::unique_ptr<LinearShear> linearShear;

  Array2D<real> Subview(Array3D<real> slice, int direction, int idx);
};

#endif // OUTPUT_SLICE_HPP_
