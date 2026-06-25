// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RIGHTHANDSIDE_BOUSSINESQ_HPP_
#define RIGHTHANDSIDE_BOUSSINESQ_HPP_

#include <memory>
#include <vector>
#include <string>
#include "rightHandSide.hpp"
#include "input.hpp"
#include "arrays.hpp"
#include "shear.hpp"
#include "hydro.hpp"
#include "mhd.hpp"

using RhsPtr = std::unique_ptr<RightHandSideConcept<Array3D<complex>>>;

class Grid;

// A class for the boussinesq right hand side
template <typename Shear>
class Boussinesq : public RightHandSide<Array3D<complex>, Shear> {
 public:
  Boussinesq(Input &input, Grid *grid, std::vector<RhsPtr> &rhsVector);
  ~Boussinesq();

  void ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) override;
  void ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) override;
  void PostStage(Field<Array3D<complex>>& fldin, real t) override;

  real GetInvDt() override;
  std::vector<std::string> GetVariables() override;

 private:
  real kappa;
  real g1, g2, g3;
  real N2;
  int kappaOrder{1};

  Array3D<real> vr1, vr2, vr3;
  Array3D<real> thr;
  Array3D<real> wr1, wr2, wr3;
  Array3D<complex> wf1, wf2, wf3;
  std::array<int,3> npr, npf;
};


// Implementation
#include "grid.hpp"
#include "loop.hpp"
#include "reduce.hpp"
#include "global.hpp"
#include "fft.hpp"

template <typename Shear>
Boussinesq<Shear>::Boussinesq(Input &input, Grid *grid, std::vector<RhsPtr> &rhsVector) :
                            RightHandSide<Array3D<complex>, Shear>(input, grid, rhsVector) {
  // Allocate all of the temporary arrays
  wr1 = astra::makeArray<Array3D<real>>("Boussinesq::wr1", grid->npr_t);
  wf1 = astra::makeArray<Array3D<complex>>("Boussinesq::wf1", grid->npf);
  wr2 = astra::makeArray<Array3D<real>>("Boussinesq::wr2", grid->npr_t);
  wf2 = astra::makeArray<Array3D<complex>>("Boussinesq::wf2", grid->npf);
  wr3 = astra::makeArray<Array3D<real>>("Boussinesq::wr3", grid->npr_t);
  wf3 = astra::makeArray<Array3D<complex>>("Boussinesq::wf3", grid->npf);
  thr = astra::makeArray<Array3D<real>>("Boussinesq::thr", grid->npr_t);

  // Set the velocity field to be used by the boussinesq class
  bool found = false;

  // walk through the rhs vector and find the hydro or mhd class, and use their velocity field as the velocity field for the boussinesq class
  bool foundHydro = false;
  for(auto &rhs : rhsVector) {
    if(dynamic_cast<Hydro<Shear>*>(rhs.get())) {
        auto hydro = dynamic_cast<Hydro<Shear>*>(rhs.get());
        this->vr1 = hydro->vr1;
        this->vr2 = hydro->vr2;
        this->vr3 = hydro->vr3;
        foundHydro = true;
        break;
    } else if(dynamic_cast<Mhd<Shear>*>(rhs.get())) {
        auto mhd = dynamic_cast<Mhd<Shear>*>(rhs.get());
        this->vr1 = mhd->vr1;
        this->vr2 = mhd->vr2;
        this->vr3 = mhd->vr3;
        foundHydro = true;
        break;
    }
  }

  if (!foundHydro) {
    throw std::runtime_error("Boussinesq right hand side requires Hydro or Mhd right hand side to be present in the rhs");
  }

  npr=grid->npr_t;
  npf=grid->npf;

  this->kappa = input.GetOrSet<real>("Physics","kappa",0,1e-3);
  this->kappaOrder = input.GetOrSet<int>("Physics","kappa",1,1);
  this->g1 = input.GetOrSet<real>("Physics","gstrat",0,0.0);
  this->g2 = input.GetOrSet<real>("Physics","gstrat",1,0.0);
  this->g3 = input.GetOrSet<real>("Physics","gstrat",2,1.0);
  // normalize the gravity vector
  real gnorm = std::sqrt(g1*g1+g2*g2+g3*g3);
  if(gnorm==0.0) {
    throw std::runtime_error("Boussinesq right hand side requires a non-zero gravity vector");
  }
  this->g1 /= gnorm;
  this->g2 /= gnorm;
  this->g3 /= gnorm;
  this->N2 = input.Get<real>("Physics","N2",0);

  astra::cout << "Boussinesq: diffusivity kappa=" << kappa << std::endl;
  astra::cout << "Boussinesq: gravity g=[" << g1 << "," << g2 << "," << g3 << "]" << std::endl;
  astra::cout << "Boussinesq: N2=" << N2 << std::endl;
}

template <typename Shear>
Boussinesq<Shear>::~Boussinesq() {
}

template <typename Shear>
void Boussinesq<Shear>::ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) {
  astra::pushRegion("Boussinesq::ExplicitStep");
  // Fourier transform the velocity field to real space (use transposed arrays)
  auto th = fldin["th"];
  // Normally vr has already been computed by the Hydro or Mhd class
  auto vr1 = this->vr1;
  auto vr2 = this->vr2;
  auto vr3 = this->vr3;

  this->grid->fft->C2R(th, thr, false);
  auto wr1 = this->wr1;
  auto wr2 = this->wr2;
  auto wr3 = this->wr3;

  auto thr = this->thr;

  astra_for("Boussinesq::HydroTransport",0,npr[IDIR],0,npr[JDIR],0,npr[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      wr1(i,j,k) = thr(i,j,k)*vr1(i,j,k);
      wr2(i,j,k) = thr(i,j,k)*vr2(i,j,k);
      wr3(i,j,k) = thr(i,j,k)*vr3(i,j,k);
  });

  // Fourier transform back to spectral space
  this->grid->fft->R2C(wr1, wf1, false);
  this->grid->fft->R2C(wr2, wf2, false);
  this->grid->fft->R2C(wr3, wf3, false);

  // Compute the cross-correlation
  auto wf1 = this->wf1;
  auto wf2 = this->wf2;
  auto wf3 = this->wf3;

  auto kx1 = this->grid->kx[IDIR];
  auto kx2 = this->grid->kx[JDIR];
  auto kx3 = this->grid->kx[KDIR];

  auto dvx1 = dfld["vx1"];
  auto dvx2 = dfld["vx2"];
  auto dvx3 = dfld["vx3"];
  auto dth = dfld["th"];
  auto v1 = fldin["vx1"];
  auto v2 = fldin["vx2"];
  auto v3 = fldin["vx3"];

  real kx1max = this->grid->kmax[IDIR];
  real kx2max = this->grid->kmax[JDIR];
  real kx3max = this->grid->kmax[KDIR];

  real g1 = this->g1;
  real g2 = this->g2;
  real g3 = this->g3;
  real N2 = this->N2;
  Shear &shear = this->shear;
  shear.Refresh(t);

  // Compute the nonlinear term in spectral space
  astra_for("hydro_nonlinear", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      // 2/3 de-aliasing rule
      complex mask = (std::fabs(kx1(i))< 2./3*kx1max
                   && std::fabs(kx2(j))< 2./3*kx2max
                   && std::fabs(kx3(k))< 2./3*kx3max) ? 1.0 : 0.0;
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));

      // dv = -N^2 [g-k(k.g)/k^2] th
      const real kdotg = kx1t*g1+kx2t*g2+kx3t*g3;
      real k2 = kx1t*kx1t+kx2t*kx2t+kx3t*kx3t;
      if(k2==0.0) k2=1.0; // avoid division by zero
      dvx1(i,j,k) -= N2*(g1-kdotg*kx1t/k2)*th(i,j,k);
      dvx2(i,j,k) -= N2*(g2-kdotg*kx2t/k2)*th(i,j,k);
      dvx3(i,j,k) -= N2*(g3-kdotg*kx3t/k2)*th(i,j,k);

      // dth = -nabla.(v th)+g.v
      dth(i,j,k) = -mask * Kokkos::complex(0.0,1.0) * (kx1t*wf1(i,j,k) + kx2t*wf2(i,j,k) + kx3t*wf3(i,j,k));
      dth(i,j,k) += g1*v1(i,j,k) + g2*v2(i,j,k) + g3*v3(i,j,k);
  });

  astra::popRegion();
}

template <typename Shear>
void Boussinesq<Shear>::ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) {
  astra::pushRegion("Boussinesq::ImplicitStep");
  auto kx1 = this->grid->kx[IDIR];
  auto kx2 = this->grid->kx[JDIR];
  auto kx3 = this->grid->kx[KDIR];

  auto th = fldin["th"];

  real kappa = this->kappa;
  int n = this->kappaOrder;
  Shear shear = this->shear;
  shear.Refresh(t);
  astra_for("boussinesq_diffusivity", fldin,
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));
      const real k2t = kx1t*kx1t + kx2t*kx2t + kx3t*kx3t;

      //real factor = (1-0.5*dt*nu*k2)/(1+0.5*dt*nu*k2); // Crank-Nicholson
      real factor = std::exp(-dt * pow(kappa*k2t, n)); // Exact integration
      th(i,j,k) *= factor;
    });
  astra::popRegion();
}

template <typename Shear> real Boussinesq<Shear>::GetInvDt() {
  astra::pushRegion("Boussinesq::GetInvDt");
  real invdt = std::sqrt(std::fabs(this->N2));

  astra::popRegion();
  return invdt;
}

template <typename Shear>
std::vector<std::string> Boussinesq<Shear>::GetVariables() {
  return {"th"};
}

template <typename Shear>
void Boussinesq<Shear>::PostStage(Field<Array3D<complex>>& fldin, real t) {
    if constexpr(Shear::isEnabled) {
    if(this->shear.NeedRemap(t)) {
      this->shear.Remap(t, fldin["th"]);

      this->shear.SetTinit(t);
    }
  }
}

#endif // RIGHTHANDSIDE_BOUSSINESQ_HPP_
