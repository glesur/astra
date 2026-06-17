// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RIGHTHANDSIDE_BURGERS_HPP_
#define RIGHTHANDSIDE_BURGERS_HPP_

#include <vector>
#include <string>
#include "rightHandSide.hpp"
#include "input.hpp"
#include "arrays.hpp"
#include "shear.hpp"

class Grid;

// A class for the burgers right hand side
template <typename Shear>
class Burgers : public RightHandSide<Array3D<complex>, Shear> {
 public:
  Burgers(Input &input, Grid *grid);
  ~Burgers();

  void ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) override;
  void ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) override;
  void PostStage(Field<Array3D<complex>>& fldin, real t) override;

  real GetInvDt() override;
  std::vector<std::string> GetVariables() override;

 private:
  real nu;
  real epsilon{0};
  real kmin{0};

  bool haveSVV{false};
  int viscosityOrder{1};
  int direction{IDIR};
  Array3D<real> vr;
  Array3D<real> wr11;
  Array3D<complex> wf11;
  std::array<int,3> npr, npf;
};


// Implementation
#include "grid.hpp"
#include "loop.hpp"
#include "reduce.hpp"
#include "global.hpp"
#include "fft.hpp"

template <typename Shear>
Burgers<Shear>::Burgers(Input &input, Grid *grid) : RightHandSide<Array3D<complex>, Shear>(input, grid) {
  // Allocate all of the temporary arrays
  vr = astra::makeArray<Array3D<real>>("Burgers::vr1", grid->npr_t);
  wr11 = astra::makeArray<Array3D<real>>("Burgers::wr11", grid->npr_t);
  wf11 = astra::makeArray<Array3D<complex>>("Burgers::wf11", grid->npf);

  npr=grid->npr_t;
  npf=grid->npf;

  this->nu = input.GetOrSet<real>("Physics","viscosity",0,1e-3);
  this->viscosityOrder = input.GetOrSet<int>("Physics","viscosity",1,1);
  this->direction = input.GetOrSet<int>("Physics","direction",0,IDIR);
  if(input.CheckEntry("Physics","svv")>0) {
    this->haveSVV = true;
    this->epsilon = input.GetOrSet<real>("Physics","svv",0,1.0);
    this->kmin = input.GetOrSet<real>("Physics","svv",1,0.2);
  }

  astra::cout << "Burgers: viscosity nu=" << nu << std::endl;
}

template <typename Shear>
Burgers<Shear>::~Burgers() {
}

template <typename Shear>
void Burgers<Shear>::ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) {
  astra::pushRegion("Burgers::ExplicitStep");
  // Fourier transform the velocity field to real space (use transposed arrays)
  auto v = fldin["v"];
  auto vr = this->vr;
  this->grid->fft->C2R(v, vr, false);
  auto wr11 = this->wr11;
  auto wf11 = this->wf11;

  // Compute the cross-correlation

  Shear &shear = this->shear;
  shear.Refresh(t);
  astra_for("hydro_windup", 0,npr[IDIR],0,npr[JDIR],0,npr[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      real v1 = vr(i, j, k);
      wr11(i, j, k) = v1 * v1 / 2.0;
    });

  // Fourier transform back to spectral space
  this->grid->fft->R2C(wr11, wf11, false);

  auto kx = this->grid->kx[direction];
  real kxmax = this->grid->kmax[direction];
  auto dv = dfld["v"];
  const int dir = this->direction;

  // Compute the nonlinear term in spectral space
  astra_for("Burger_nonlinear", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      // 2/3 de-aliasing rule
      real kdir = 0;
      if(dir==IDIR) kdir = kx(i);
      else if(dir==JDIR) kdir = kx(j);
      else if(dir==KDIR) kdir = kx(k);

      // 2/3 de-aliasing rule
      complex mask = (std::fabs(kdir)< 2./3*kxmax) ? 1.0 : 0.0;

      // dv = - ik wf11
      dv(i,j,k) -= Kokkos::complex(0.0,1.0) * (kdir*wf11(i,j,k)) * mask;
  });

  astra::popRegion();
}

template <typename Shear>
void Burgers<Shear>::ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) {
  astra::pushRegion("Burgers::ImplicitStep");
  auto kx = this->grid->kx[direction];
  real kxmax = this->grid->kmax[direction];
  auto v = fldin["v"];

  const int dir = this->direction;
  real nu= this->nu;
  int n = this->viscosityOrder;

  bool haveSVV = this->haveSVV;
  real epsilon = this->epsilon;
  real kmin = this->kmin;
  // Compute the nonlinear term in spectral space
  astra_for("Burger_nonlinear", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      // 2/3 de-aliasing rule
      real kdir = 0;
      if(dir==IDIR) kdir = kx(i);
      else if(dir==JDIR) kdir = kx(j);
      else if(dir==KDIR) kdir = kx(k);
      // Standard laplacian viscosity
      real factor = std::exp(-dt * pow(nu*kdir*kdir, n)); // Exact integration

      // Artificial viscosity for high wavenumbers (SVV)
      if(haveSVV) {
        if(std::fabs(kdir) > kmin*kxmax) {
          real coeff = std::pow(kdir-kxmax,2)/std::pow(kdir-kmin*kxmax,2);
          real nucvv = std::exp(-coeff);
          factor *= std::exp(-dt * epsilon * kdir*kdir * nucvv);
        }
      }
      v(i,j,k) *= factor;
    });
  astra::popRegion();
}

template <typename Shear>
real Burgers<Shear>::GetInvDt() {
  astra::pushRegion("Burgers::GetInvDt");
  real invdt = 0.0;
  real kxmax = this->grid->kmax[direction];

  auto vr = this->vr;

  astra_reduce("Timestep_reduction",
    0,npr[IDIR],
    0,npr[JDIR],
    0,npr[KDIR],
    KOKKOS_LAMBDA (int64_t i, int64_t j, int64_t k, real &dtmax) {
      real gamma_v = std::fabs(vr(i,j,k)*kxmax);

      dtmax = std::fmax(dtmax, gamma_v);
        },
    Kokkos::Max<real>(invdt));

  #ifdef WITH_MPI
    // Across all MPI processes
    MPI_Allreduce(MPI_IN_PLACE, &invdt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  #endif
  astra::popRegion();
  return invdt;
}

template <typename Shear>
std::vector<std::string> Burgers<Shear>::GetVariables() {
  return {"v"};
}

template <typename Shear>
void Burgers<Shear>::PostStage(Field<Array3D<complex>>& fldin, real t) {
}

#endif // RIGHTHANDSIDE_BURGERS_HPP_
