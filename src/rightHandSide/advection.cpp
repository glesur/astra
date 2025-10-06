// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A purely advective right hand side along a given direction

#include "advection.hpp"
#include "astra.hpp"
#include "arrays.hpp"
#include "input.hpp"
#include "rightHandSide.hpp"
#include "loop.hpp"
#include "grid.hpp"
#include "global.hpp"
#include "loop.hpp"
#include "fft.hpp"
#include "vtk.hpp"

Advection::Advection(Input &input, Grid *grid) : RightHandSide<Array3D<complex>>(input, grid) {
  direction = input.Get<int>("Advection","direction",0);
  velocity = input.GetOrSet<real>("Advection","velocity", 0,1.0);
  if(direction < 0 || direction > 2) {
    throw std::runtime_error("Advection direction must be 0, 1 or 2");
  }
  astra::cout << "Advection along direction " << direction << " with velocity " << velocity << std::endl;
}

Advection::~Advection() {}

void Advection::ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) {
  astra::pushRegion("Advection::ExplicitStep");
  // Compute the advection term
    static int vtk_counter_in = 0;
  Vtk vtk(grid, t, "debug_advection_fldin"+std::to_string(vtk_counter_in));
  vtk.Write(fldin);
  vtk_counter_in++;
  
  for(auto& it : fldin) {
    auto view = it.second;
    auto dview = dfld[it.first];
    auto kx = this->grid->kx[direction];
    int direction = this->direction;
    real velocity = this->velocity;
    

    

    astra_for("advection_"+it.first,fldin,
      KOKKOS_LAMBDA(int i,int j,int k) {
        complex kv;
        if(direction == 0) {
          kv = kx(i)*velocity;
        } else if(direction == 1) {
          kv = kx(j)*velocity;
        } else {
          kv = kx(k)*velocity;
        }
        //kv = 1.0;
        dview(i,j,k) -= Kokkos::complex(0.0, 1.0)*kv*view(i,j,k);
        //dview(i,j,k) = view(i,j,k);
    });
  }
    static int vtk_counter_out = 0;
  Vtk vtko(grid, t, "debug_advection_fldout"+std::to_string(vtk_counter_out));
  vtko.Write(dfld);
  vtk_counter_out++;
  astra::popRegion();
}

void Advection::ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt){
  // Nothing to do here for pure advection
}

real Advection::GetInvDt()  {
  astra::pushRegion("Advection::GetInvDt");
  real invdt = this->velocity*grid->kmax[direction];
  astra::popRegion();
  return invdt;
}

std::vector<std::string> Advection::GetVariables() {
  return {"rho"};
}