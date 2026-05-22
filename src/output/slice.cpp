// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <string>
#include <map>
#include <memory>
#include "slice.hpp"
#include "subgrid.hpp"
#include "input.hpp"
#include "vtk.hpp"
#include "dumpVariables.hpp"
#include "output.hpp"

Slice::Slice() = default;

Slice::Slice(Input &input, Grid *grid, int nSlice) {
  this->input = &input;
  std::string prefix = "vtk_slice"+std::to_string(nSlice);
  this->slicePeriod = input.Get<real>("Output", prefix,0);
  this->direction = input.Get<int>("Output", prefix,1);
  this->x0 = input.Get<real>("Output", prefix, 2);
  this->nSlice = nSlice;

  if(input.CheckEntry("Output","vtk_dir")>=0) {
    outputVtkDirectory = input.Get<std::string>("Output","vtk_dir",0);
  }

  std::string typeStr = input.Get<std::string>("Output",prefix,3);
  if(typeStr.compare("cut")==0) {
    type = SliceType::Cut;
  } else if(typeStr.compare("average")==0) {
    type = SliceType::Average;
  } else {
    throw std::runtime_error("Unknown slice type "+typeStr);
  }

  if(slicePeriod> 0) {
    sliceLast = - slicePeriod; // so that output at time 0.0
  } else {
    sliceLast = 0.0;
  }
  // Register the last output in dumps so that we restart from the right slice
  DumpVariables::Register(std::string("slcLast-")+std::to_string(nSlice), sliceLast);
  DumpVariables::Register(std::string("slcNvtk-")+std::to_string(nSlice), nvtk);

  // Initialize the subGrid
  try {
    this->subGrid = std::make_unique<SubGrid>(grid, type, direction, x0);
  } catch(const std::exception& e) {
    std::stringstream msg;
    msg << "Slice: Error while initializing vtk slice " << nSlice << ": " << e.what() << std::endl;
    throw std::runtime_error(msg.str());
  }
  this->subGrid = std::make_unique<SubGrid>(grid, type, direction, x0);
  this->containsX0 = subGrid->containsX0;

  // Shear when needed
  std::string shearTypeStr = input.GetOrSet<std::string>("Physics","shear_type",0,"disabled");
  if(shearTypeStr =="linear") {
    this->haveShear = true;
    this->linearShear = std::make_unique<LinearShear>(input, grid);
  }

  #ifdef WITH_MPI
    if(type==SliceType::Average) {
      // Create a communicator on which we can do the sum accross processors
    }
  #endif
}

// A workaround to ensure that 2D views are kokkos::deep_copyable, as subviews of 3D views are not directly copyable to the host
// i.e. error "There is no common execution space that can access both source's space"
Array2D<real> Slice::Subview(Array3D<real> array, int direction, int idx) {
  Array2D<real> slice;
  if(direction == IDIR) {
    auto subview = Kokkos::subview(array, idx, Kokkos::ALL(), Kokkos::ALL());
    slice = Array2D<real>("slice", subview.extent(0), subview.extent(1));
    Kokkos::deep_copy(slice, subview);
  } else if(direction == JDIR) {
    auto subview = Kokkos::subview(array, Kokkos::ALL(), idx, Kokkos::ALL());
    slice = Array2D<real>("slice", subview.extent(0), subview.extent(1));
    Kokkos::deep_copy(slice, subview);
  } else if(direction == KDIR) {
    auto subview = Kokkos::subview(array, Kokkos::ALL(), Kokkos::ALL(), idx);
    slice = Array2D<real>("slice", subview.extent(0), subview.extent(1));
    Kokkos::deep_copy(slice, subview);
  }

  return slice;
}
void Slice::WriteSlice(Array3D<real> slice, Vtk *vtk, real time, std::string name) {
  if(this->type == SliceType::Cut && containsX0) {
      // index of element in current subgrid corresponding to the slice position
      const int idx = subGrid->index;
      auto myslice = this->Subview(slice, direction, idx);
      vtk->Write(myslice, time, name);
    }
    if(this->type == SliceType::Average) {
      // TO BE DONE
    }
}


void Slice::CheckForWrite(Field<Array3D<complex>> state, real time, Output *output) {
  if(slicePeriod>=0.0 && time-sliceLast>=slicePeriod) {
    std::stringstream ssfileName, ssvtkFileNum;
    ssvtkFileNum << std::setfill('0') << std::setw(4) << nvtk;
    std::string filename = std::string("slice") + std::to_string(nSlice) + "." + ssvtkFileNum.str();

    //
    Vtk *vtk;
    if(containsX0) {
      vtk = new Vtk(subGrid.get(), *input, time, filename,outputVtkDirectory);
    }

    Array3D<real> realView = astra::makeArray<Array3D<real>>("real_array", subGrid->parentGrid->npr);

    // Write the slice for each variable in the field
    for(auto const &it : state) {
      auto name = it.first;
      auto &view = it.second;
      // Fourier transform
      subGrid->parentGrid->fft->C2R(view, realView);
      Kokkos::fence();
      if(haveShear) {
        this->linearShear->SetTinit(time);
        this->linearShear->UnshearFrame(realView);
      }
      WriteSlice(realView, vtk, time, name);
    }

    // Additional variables output
    if(!output->vtkAdditionalVariableNames.empty()) {
      output->ComputeAdditionalVariables(state, time);
      for(auto const &it : output->vtkAdditionalVariables) {
        auto name = it.first;
        auto &view = it.second;
        // Fourier transform
        subGrid->parentGrid->fft->C2R(view, realView);
        Kokkos::fence();
        if(haveShear) {
          this->linearShear->SetTinit(time);
          this->linearShear->UnshearFrame(realView);
        }
        WriteSlice(realView, vtk, time, name);
      }
    }
    if(containsX0) {
      delete vtk; // Clean up the VTK object
    }
    nvtk++;
    while(time-sliceLast>=slicePeriod) {
      sliceLast += slicePeriod;
    }
  }
}
