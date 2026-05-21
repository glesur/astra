// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A purely advective right hand side along a given direction

#ifndef RIGHTHANDSIDE_ADVECTION_HPP_
#define RIGHTHANDSIDE_ADVECTION_HPP_

#include <string>
#include <vector>
#include "rightHandSide.hpp"
#include "input.hpp"
#include "arrays.hpp"

class Grid;
template <typename Shear>
class Advection : public RightHandSide<Array3D<complex>, Shear> {
 public:
  Advection(Input &input, Grid *grid);

  ~Advection();

  void ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) override;
  void ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) override;
  void PostStage(Field<Array3D<complex>>& fldin, real t) override {};

  real GetInvDt() override;

  std::vector<std::string> GetVariables() override;

 private:
  int direction{0};
  real velocity{1.0};
};

// Implementation
#include "astra.hpp"
#include "grid.hpp"
#include "loop.hpp"
#include "global.hpp"
#include "fft.hpp"

template <typename Shear>
Advection<Shear>::Advection(Input &input, Grid *grid) : RightHandSide<Array3D<complex>, Shear>(input, grid) {
  direction = input.Get<int>("Physics","direction",0);
  velocity = input.GetOrSet<real>("Physics","velocity", 0,1.0);
  if(direction < 0 || direction > 2) {
    throw std::runtime_error("Advection direction must be 0, 1 or 2");
  }
  astra::cout << "Advection along direction " << direction << " with velocity " << velocity << std::endl;
}

template <typename Shear>
Advection<Shear>::~Advection() {}

template <typename Shear>
void Advection<Shear>::ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) {
  astra::pushRegion("Advection::ExplicitStep");

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
        dview(i,j,k) -= Kokkos::complex(0.0, 1.0)*kv*view(i,j,k);
    });
  }
  astra::popRegion();
}

template <typename Shear>
void Advection<Shear>::ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) {
  // Nothing to do here for pure advection
}

template <typename Shear>
real Advection<Shear>::GetInvDt()  {
  astra::pushRegion("Advection::GetInvDt");
  real invdt = this->velocity*this->grid->kmax[direction];
  astra::popRegion();
  return invdt;
}

template <typename Shear>
std::vector<std::string> Advection<Shear>::GetVariables() {
  return {"rho"};
}
#endif // RIGHTHANDSIDE_ADVECTION_HPP_
