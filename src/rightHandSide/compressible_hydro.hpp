// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RIGHTHANDSIDE_COMPRESSIBLE_HYDRO_HPP_
#define RIGHTHANDSIDE_COMPRESSIBLE_HYDRO_HPP_

#include <memory>
#include <vector>
#include <string>
#include "rightHandSide.hpp"
#include "input.hpp"
#include "arrays.hpp"
#include "shear.hpp"

using RhsPtr = std::unique_ptr<RightHandSideConcept<Array3D<complex>>>;

class Grid;

// A class for the hydrodynamics right hand side
template <typename Shear>
class CompressibleHydro : public RightHandSide<Array3D<complex>, Shear> {
 public:
  CompressibleHydro(Input &input, Grid *grid, std::vector<RhsPtr> &rhsVector);
  ~CompressibleHydro();

  void ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) override;
  void ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) override;
  void PostStage(Field<Array3D<complex>>& fldin, real t) override;
  real GetInvDt() override;
  std::vector<std::string> GetVariables() override;

 private:
  real nu;
  real etaRho;
  int viscosityOrder{1};
  int etaRhoOrder{1};
  real rhoFloor;
  bool haveSourceTerm{false};
  real Omega;
  real cs{1.0}; // Sound speed (for isothermal equation of state)
  Array3D<real> pr1, pr2, pr3;
  //Array3D<real> vr1,vr2,vr3;
  Array3D<real> rhor;
  Array3D<real> wr11,wr12,wr13,wr22,wr23,wr33;
  Array3D<complex> wf11,wf12,wf13,wf22,wf23,wf33;
  std::array<int,3> npr, npf;
};


// Implementation
#include "grid.hpp"
#include "loop.hpp"
#include "reduce.hpp"
#include "global.hpp"
#include "fft.hpp"

template <typename Shear>
CompressibleHydro<Shear>::CompressibleHydro(Input &input, Grid *grid, std::vector<RhsPtr> &rhsVector) :
                                            RightHandSide<Array3D<complex>, Shear>(input, grid, rhsVector) {
  // Allocate all of the temporary arrays
  //vr1 = astra::makeArray<Array3D<real>>("CompressibleHydro::vr1", grid->npr_t);
  //vr2 = astra::makeArray<Array3D<real>>("CompressibleHydro::vr2", grid->npr_t);
  //vr3 = astra::makeArray<Array3D<real>>("CompressibleHydro::vr3", grid->npr_t);
  pr1 = astra::makeArray<Array3D<real>>("CompressibleHydro::pr1", grid->npr_t);
  pr2 = astra::makeArray<Array3D<real>>("CompressibleHydro::pr2", grid->npr_t);
  pr3 = astra::makeArray<Array3D<real>>("CompressibleHydro::pr3", grid->npr_t);
  rhor = astra::makeArray<Array3D<real>>("CompressibleHydro::rhor", grid->npr_t);
  wr11 = astra::makeArray<Array3D<real>>("CompressibleHydro::wr11", grid->npr_t);
  wr12 = astra::makeArray<Array3D<real>>("CompressibleHydro::wr12", grid->npr_t);
  wr13 = astra::makeArray<Array3D<real>>("CompressibleHydro::wr13", grid->npr_t);
  wr22 = astra::makeArray<Array3D<real>>("CompressibleHydro::wr22", grid->npr_t);
  wr23 = astra::makeArray<Array3D<real>>("CompressibleHydro::wr23", grid->npr_t);
  wr33 = astra::makeArray<Array3D<real>>("CompressibleHydro::wr33", grid->npr_t);
  wf11 = astra::makeArray<Array3D<complex>>("CompressibleHydro::wf11", grid->npf);
  wf12 = astra::makeArray<Array3D<complex>>("CompressibleHydro::wf12", grid->npf);
  wf13 = astra::makeArray<Array3D<complex>>("CompressibleHydro::wf13", grid->npf);
  wf22 = astra::makeArray<Array3D<complex>>("CompressibleHydro::wf22", grid->npf);
  wf23 = astra::makeArray<Array3D<complex>>("CompressibleHydro::wf23", grid->npf);
  wf33 = astra::makeArray<Array3D<complex>>("CompressibleHydro::wf33", grid->npf);
  npr=grid->npr_t;
  npf=grid->npf;

  this->nu = input.GetOrSet<real>("Physics","viscosity",0,1e-3);
  this->viscosityOrder = input.GetOrSet<int>("Physics","viscosity",1,1);

  this->etaRho = input.GetOrSet<real>("Physics","eta_rho",0,0.0);
  this->etaRhoOrder = input.GetOrSet<int>("Physics","eta_rho",1,1);

  this->rhoFloor = input.GetOrSet<real>("Physics","rho_floor",0,1e-6);

  this->Omega = input.GetOrSet<real>("Physics","omega",0,0.0);
  this->haveSourceTerm = (input.CheckEntry("Physics","omega") >0 || Shear::isEnabled);
  this->cs = input.GetOrSet<real>("Physics","cs",0,1.0);

  astra::cout << "CompressibleHydro: viscosity nu=" << nu << std::endl;
  astra::cout << "CompressibleHydro: sound speed cs=" << cs << std::endl;
  if(haveSourceTerm) {
    astra::cout << "CompressibleHydro: Omega=" << Omega << " and shear rate S=" << this->shear.shearRate << std::endl;
  }
}

template <typename Shear>
CompressibleHydro<Shear>::~CompressibleHydro() {
}

template <typename Shear>
void CompressibleHydro<Shear>::ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) {
  astra::pushRegion("CompressibleHydro::ExplicitStep");
  // Fourier transform the velocity field to real space (use transposed arrays)

  auto px1 = fldin["px1"];
  auto px2 = fldin["px2"];
  auto px3 = fldin["px3"];
  auto rho = fldin["rho"];

  this->grid->fft->C2R(px1, pr1, false);
  this->grid->fft->C2R(px2, pr2, false);
  this->grid->fft->C2R(px3, pr3, false);
  this->grid->fft->C2R(rho, rhor, false);

  // Compute the cross-correlation
  auto pr1 = this->pr1;
  auto pr2 = this->pr2;
  auto pr3 = this->pr3;
  auto rhor = this->rhor;
  auto wr11 = this->wr11;
  auto wr12 = this->wr12;
  auto wr13 = this->wr13;
  auto wr22 = this->wr22;
  auto wr23 = this->wr23;
  auto wr33 = this->wr33;

  Shear &shear = this->shear;
  shear.Refresh(t);

  real rhoFloor = this->rhoFloor;
  astra_for("hydro_windup", 0,npr[IDIR],0,npr[JDIR],0,npr[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      real p1 = pr1(i, j, k);
      real p2 = pr2(i, j, k);
      real p3 = pr3(i, j, k);

      const real rhor_val = std::fmax(rhor(i, j, k), rhoFloor);  // avoid division by ~0 when computing p_i p_j / rho
      wr11(i, j, k) = p1 * p1 / rhor_val;
      wr12(i, j, k) = p1 * p2 / rhor_val;
      wr13(i, j, k) = p1 * p3 / rhor_val;
      wr22(i, j, k) = p2 * p2 / rhor_val;
      wr23(i, j, k) = p2 * p3 / rhor_val;
      wr33(i, j, k) = p3 * p3 / rhor_val;
    });

  // Fourier transform back to spectral space
  this->grid->fft->R2C(wr11, wf11, false);
  this->grid->fft->R2C(wr12, wf12, false);
  this->grid->fft->R2C(wr13, wf13, false);
  this->grid->fft->R2C(wr22, wf22, false);
  this->grid->fft->R2C(wr23, wf23, false);
  this->grid->fft->R2C(wr33, wf33, false);

  auto kx1 = this->grid->kx[IDIR];
  auto kx2 = this->grid->kx[JDIR];
  auto kx3 = this->grid->kx[KDIR];

  auto dpx1 = dfld["px1"];
  auto dpx2 = dfld["px2"];
  auto dpx3 = dfld["px3"];
  auto wf11 = this->wf11;
  auto wf12 = this->wf12;
  auto wf13 = this->wf13;
  auto wf22 = this->wf22;
  auto wf23 = this->wf23;
  auto wf33 = this->wf33;

  real kx1max = this->grid->kmax[IDIR];
  real kx2max = this->grid->kmax[JDIR];
  real kx3max = this->grid->kmax[KDIR];
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
      dpx1(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1t*wf11(i,j,k)+kx2t*wf12(i,j,k)+kx3t*wf13(i,j,k))*mask;
      dpx2(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1t*wf12(i,j,k)+kx2t*wf22(i,j,k)+kx3t*wf23(i,j,k))*mask;
      dpx3(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1t*wf13(i,j,k)+kx2t*wf23(i,j,k)+kx3t*wf33(i,j,k))*mask;
  });

  // Source terms
  if(haveSourceTerm) {
    real Omega = this->Omega;
    real S = this->shear.shearRate;
    astra_for("hydro_source_terms", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
      KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
        dpx1(i,j,k) += 2.0*Omega*px2(i,j,k);
        dpx2(i,j,k) += -(2.0*Omega - S)*px1(i,j,k);
    });
  }
  // Pressure term and continuity equation
  auto drho = dfld["rho"];

  const real cs2 = this->cs*this->cs;
  astra_for("hydro_pressure", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));

      dpx1(i,j,k) -= cs2 * Kokkos::complex(0.0,1.0) * kx1t * rho(i,j,k);
      dpx2(i,j,k) -= cs2 * Kokkos::complex(0.0,1.0) * kx2t * rho(i,j,k);
      dpx3(i,j,k) -= cs2 * Kokkos::complex(0.0,1.0) * kx3t * rho(i,j,k);

      drho(i,j,k) -= Kokkos::complex(0.0,1.0) * (kx1t*px1(i,j,k)+kx2t*px2(i,j,k)+kx3t*px3(i,j,k));
  });
  astra::popRegion();
}


template <typename Shear>
void CompressibleHydro<Shear>::ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) {
  astra::pushRegion("CompressibleHydro::ImplicitStep");
  auto kx1 = this->grid->kx[IDIR];
  auto kx2 = this->grid->kx[JDIR];
  auto kx3 = this->grid->kx[KDIR];

  auto px1 = fldin["px1"];
  auto px2 = fldin["px2"];
  auto px3 = fldin["px3"];
  auto rho = fldin["rho"];

  real nu= this->nu;
  int n = this->viscosityOrder;
  real etaRho = this->etaRho;
  int nRho = this->etaRhoOrder;

  Shear shear = this->shear;
  shear.Refresh(t);
  astra_for("CompressibleHydro_viscosity", fldin,
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));
      const real k2t = kx1t*kx1t + kx2t*kx2t + kx3t*kx3t;

      //real factor = (1-0.5*dt*nu*k2)/(1+0.5*dt*nu*k2); // Crank-Nicholson
      real factor = std::exp(-dt * pow(nu*k2t, n)); // Exact integration
      px1(i,j,k) *= factor;
      px2(i,j,k) *= factor;
      px3(i,j,k) *= factor;

      real factorRho = std::exp(-dt * pow(etaRho*k2t, nRho)); // Exact integration
      rho(i,j,k) *= factorRho;
    });
  astra::popRegion();
}

template <typename Shear>
real CompressibleHydro<Shear>::GetInvDt() {
  astra::pushRegion("CompressibleHydro::GetInvDt");
  real invdt = 0.0;
  real kx1max = this->shear.kx1tmax();
  real kx2max = this->shear.kx2tmax();
  real kx3max = this->shear.kx3tmax();

  auto pr1 = this->pr1;
  auto pr2 = this->pr2;
  auto pr3 = this->pr3;
  auto rhor = this->rhor;
  real Omega = std::fabs(this->Omega);
  real S = std::fabs(this->shear.shearRate);
  real cs = this->cs;
  astra_reduce("Timestep_reduction",
    0,npr[IDIR],
    0,npr[JDIR],
    0,npr[KDIR],
    KOKKOS_LAMBDA (int64_t i, int64_t j, int64_t k, real &dtmax) {
      real v1 = std::fabs(pr1(i,j,k))/rhor(i,j,k);
      real v2 = std::fabs(pr2(i,j,k))/rhor(i,j,k);
      real v3 = std::fabs(pr3(i,j,k))/rhor(i,j,k);
      real idtx1 = (v1+cs)*kx1max;
      real idtx2 = (v2+cs)*kx2max;
      real idtx3 = (v3+cs)*kx3max;

      real gamma_v = idtx1+idtx2+idtx3;
      gamma_v += Omega; // rotation source term
      gamma_v += S; // shear source term
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
std::vector<std::string> CompressibleHydro<Shear>::GetVariables() {
  return {"rho","px1","px2","px3"};
}

template <typename Shear>
void CompressibleHydro<Shear>::PostStage(Field<Array3D<complex>>& fldin, real t) {
  if constexpr(Shear::isEnabled) {
    if(this->shear.NeedRemap(t)) {
      this->shear.Remap(t, fldin["px1"]);
      this->shear.Remap(t, fldin["px2"]);
      this->shear.Remap(t, fldin["px3"]);
      this->shear.Remap(t, fldin["rho"]);

      this->shear.SetTinit(t);
    }
  }
}

#endif // RIGHTHANDSIDE_COMPRESSIBLE_HYDRO_HPP_
