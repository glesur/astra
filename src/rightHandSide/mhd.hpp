// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RIGHTHANDSIDE_MHD_HPP_
#define RIGHTHANDSIDE_MHD_HPP_

#include <memory>
#include <string>
#include <vector>
#include "field.hpp"
#include "input.hpp"
#include "rightHandSide.hpp"
#include "arrays.hpp"
#include "shear.hpp"

using RhsPtr = std::unique_ptr<RightHandSideConcept<Array3D<complex>>>;

class Grid;
template <typename S> class Boussinesq;

// A class for the hydrodynamics right hand side
template <typename Shear>
class Mhd : public RightHandSide<Array3D<complex>, Shear> {
 public:
  Mhd(Input &input, Grid *grid, std::vector<RhsPtr> &rhsVector);
  ~Mhd();

  void ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) override;
  void ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) override;
  void PostStage(Field<Array3D<complex>>& fldin, real t) override;
  void Projector(Field<Array3D<complex>>& fldin, real t);

  real GetInvDt() override;
  std::vector<std::string> GetVariables() override;

 private:
  real nu;
  real eta;
  int viscosityOrder{1};
  int resistivityOrder{1};
  bool haveSourceTerm{false};
  real Omega;
  Array3D<real> vr1, vr2, vr3;
  Array3D<real> br1, br2, br3;
  Array3D<real> wr11,wr12,wr13,wr22,wr23,wr33;
  Array3D<complex> wf11,wf12,wf13,wf22,wf23,wf33;
  std::array<int,3> npr, npf;

  friend class Boussinesq<Shear>;
};


// Implementation
#include "grid.hpp"
#include "loop.hpp"
#include "reduce.hpp"
#include "global.hpp"
#include "fft.hpp"

template <typename Shear>
Mhd<Shear>::Mhd(Input &input, Grid *grid, std::vector<RhsPtr> &rhsVector) :
                RightHandSide<Array3D<complex>, Shear>(input, grid, rhsVector) {
  // Allocate all of the temporary arrays
  vr1 = astra::makeArray<Array3D<real>>("Mhd::vr1", grid->npr_t);
  vr2 = astra::makeArray<Array3D<real>>("Mhd::vr2", grid->npr_t);
  vr3 = astra::makeArray<Array3D<real>>("Mhd::vr3", grid->npr_t);
  br1 = astra::makeArray<Array3D<real>>("Mhd::br1", grid->npr_t);
  br2 = astra::makeArray<Array3D<real>>("Mhd::br2", grid->npr_t);
  br3 = astra::makeArray<Array3D<real>>("Mhd::br3", grid->npr_t);
  wr11 = astra::makeArray<Array3D<real>>("Mhd::wr11", grid->npr_t);
  wr12 = astra::makeArray<Array3D<real>>("Mhd::wr12", grid->npr_t);
  wr13 = astra::makeArray<Array3D<real>>("Mhd::wr13", grid->npr_t);
  wr22 = astra::makeArray<Array3D<real>>("Mhd::wr22", grid->npr_t);
  wr23 = astra::makeArray<Array3D<real>>("Mhd::wr23", grid->npr_t);
  wr33 = astra::makeArray<Array3D<real>>("Mhd::wr33", grid->npr_t);
  wf11 = astra::makeArray<Array3D<complex>>("Mhd::wf11", grid->npf);
  wf12 = astra::makeArray<Array3D<complex>>("Mhd::wf12", grid->npf);
  wf13 = astra::makeArray<Array3D<complex>>("Mhd::wf13", grid->npf);
  wf22 = astra::makeArray<Array3D<complex>>("Mhd::wf22", grid->npf);
  wf23 = astra::makeArray<Array3D<complex>>("Mhd::wf23", grid->npf);
  wf33 = astra::makeArray<Array3D<complex>>("Mhd::wf33", grid->npf);
  npr=grid->npr_t;
  npf=grid->npf;

  this->nu = input.GetOrSet<real>("Physics","viscosity",0,1e-3);
  this->eta = input.GetOrSet<real>("Physics","resistivity",0,1e-3);
  this->viscosityOrder = input.GetOrSet<int>("Physics","viscosity",1,1);
  this->resistivityOrder = input.GetOrSet<int>("Physics","resistivity",1,1);
  this->Omega = input.GetOrSet<real>("Physics","omega",0,0.0);
  this->haveSourceTerm = (input.CheckEntry("Physics","omega") >0 || Shear::isEnabled);

  astra::cout << "Mhd: viscosity nu=" << nu << std::endl;
  astra::cout << "Mhd: resistivity eta=" << eta << std::endl;
  if(haveSourceTerm) {
    astra::cout << "Mhd: Omega=" << Omega << " and shear rate S=" << this->shear.shearRate << std::endl;
  }
}

template <typename Shear>
Mhd<Shear>::~Mhd() {
}

template <typename Shear>
void Mhd<Shear>::ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) {
  astra::pushRegion("Mhd::ExplicitStep");
  // Fourier transform the velocity  and magnetic fields to real space (use transposed arrays)
  auto vx1 = fldin["vx1"];
  auto vx2 = fldin["vx2"];
  auto vx3 = fldin["vx3"];
  auto bx1 = fldin["bx1"];
  auto bx2 = fldin["bx2"];
  auto bx3 = fldin["bx3"];

  this->grid->fft->C2R(vx1, vr1, false);
  this->grid->fft->C2R(vx2, vr2, false);
  this->grid->fft->C2R(vx3, vr3, false);
  this->grid->fft->C2R(bx1, br1, false);
  this->grid->fft->C2R(bx2, br2, false);
  this->grid->fft->C2R(bx3, br3, false);

  // Compute the cross-correlation
  auto vr1 = this->vr1;
  auto vr2 = this->vr2;
  auto vr3 = this->vr3;
  auto br1 = this->br1;
  auto br2 = this->br2;
  auto br3 = this->br3;
  auto wr11 = this->wr11;
  auto wr12 = this->wr12;
  auto wr13 = this->wr13;
  auto wr22 = this->wr22;
  auto wr23 = this->wr23;
  auto wr33 = this->wr33;

  Shear &shear = this->shear;
  shear.Refresh(t);
  astra_for("hydro_windup", 0,npr[IDIR],0,npr[JDIR],0,npr[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      real v1 = vr1(i, j, k);
      real v2 = vr2(i, j, k);
      real v3 = vr3(i, j, k);
      real b1 = br1(i, j, k);
      real b2 = br2(i, j, k);
      real b3 = br3(i, j, k);
      wr11(i, j, k) = v1 * v1 - b1 * b1;
      wr12(i, j, k) = v1 * v2 - b1 * b2;
      wr13(i, j, k) = v1 * v3 - b1 * b3;
      wr22(i, j, k) = v2 * v2 - b2 * b2;
      wr23(i, j, k) = v2 * v3 - b2 * b3;
      wr33(i, j, k) = v3 * v3 - b3 * b3;
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

  auto dvx1 = dfld["vx1"];
  auto dvx2 = dfld["vx2"];
  auto dvx3 = dfld["vx3"];
  auto wf11 = this->wf11;
  auto wf12 = this->wf12;
  auto wf13 = this->wf13;
  auto wf22 = this->wf22;
  auto wf23 = this->wf23;
  auto wf33 = this->wf33;

  real kx1max = this->grid->kmax[IDIR];
  real kx2max = this->grid->kmax[JDIR];
  real kx3max = this->grid->kmax[KDIR];
  // Compute the nonlinear advection term in spectral space
  astra_for("hydro_nonlinear", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      // 2/3 de-aliasing rule
      complex mask = (std::fabs(kx1(i))< 2./3*kx1max
                   && std::fabs(kx2(j))< 2./3*kx2max
                   && std::fabs(kx3(k))< 2./3*kx3max) ? 1.0 : 0.0;
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));
      dvx1(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1t*wf11(i,j,k)+kx2t*wf12(i,j,k)+kx3t*wf13(i,j,k))*mask;
      dvx2(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1t*wf12(i,j,k)+kx2t*wf22(i,j,k)+kx3t*wf23(i,j,k))*mask;
      dvx3(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1t*wf13(i,j,k)+kx2t*wf23(i,j,k)+kx3t*wf33(i,j,k))*mask;
  });

  // Magnetic field induction
  // vxB in real space
  astra_for("mhd_emf", 0,npr[IDIR],0,npr[JDIR],0,npr[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      real v1 = vr1(i, j, k);
      real v2 = vr2(i, j, k);
      real v3 = vr3(i, j, k);
      real b1 = br1(i, j, k);
      real b2 = br2(i, j, k);
      real b3 = br3(i, j, k);
      wr11(i, j, k) = v2 * b3 - v3 * b2;
      wr12(i, j, k) = v3 * b1 - v1 * b3;
      wr13(i, j, k) = v1 * b2 - v2 * b1;
    });

  // Fourier transform back to spectral space
  this->grid->fft->R2C(wr11, wf11, false);
  this->grid->fft->R2C(wr12, wf12, false);
  this->grid->fft->R2C(wr13, wf13, false);

  auto dbx1 = dfld["bx1"];
  auto dbx2 = dfld["bx2"];
  auto dbx3 = dfld["bx3"];
  // Compute the induction term in spectral space
  astra_for("mhd_induction", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      // 2/3 de-aliasing rule
      complex mask = (std::fabs(kx1(i))< 2./3*kx1max
                   && std::fabs(kx2(j))< 2./3*kx2max
                   && std::fabs(kx3(k))< 2./3*kx3max) ? 1.0 : 0.0;
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));
      dbx1(i,j,k) = Kokkos::complex(0.0,1.0)*(kx2t*wf13(i,j,k)-kx3t*wf12(i,j,k))*mask;
      dbx2(i,j,k) = Kokkos::complex(0.0,1.0)*(kx3t*wf11(i,j,k)-kx1t*wf13(i,j,k))*mask;
      dbx3(i,j,k) = Kokkos::complex(0.0,1.0)*(kx1t*wf12(i,j,k)-kx2t*wf11(i,j,k))*mask;
  });

  // Source terms
  if(haveSourceTerm) {
    real Omega = this->Omega;
    real S = this->shear.shearRate;
    astra_for("hydro_source_terms", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
      KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
        dvx1(i,j,k) += 2.0*Omega*vx2(i,j,k);
        dvx2(i,j,k) += -(2.0*Omega - S)*vx1(i,j,k);
        dbx2(i,j,k) += - S*bx1(i,j,k);
    });
  }
  // Pressure term
  astra_for("hydro_pressure", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));
      const real k2t = kx1t*kx1t + kx2t*kx2t + kx3t*kx3t;
      if(k2t > 0.0) {
        complex kv_dot_v = kx1t*dvx1(i,j,k) + kx2t*dvx2(i,j,k) + kx3t*dvx3(i,j,k);
        kv_dot_v += shear.shearRate * kx2(j) * vx1(i,j,k); // Shear contribution = dk/dt.v
        dvx1(i,j,k) -= kv_dot_v*kx1t/k2t;
        dvx2(i,j,k) -= kv_dot_v*kx2t/k2t;
        dvx3(i,j,k) -= kv_dot_v*kx3t/k2t;
      }
  });
  astra::popRegion();
}

template <typename Shear>
void Mhd<Shear>::Projector(Field<Array3D<complex>>& fldin, real t) {
  astra::pushRegion("Mhd::Projector");
  auto kx1 = this->grid->kx[IDIR];
  auto kx2 = this->grid->kx[JDIR];
  auto kx3 = this->grid->kx[KDIR];

  auto vx1 = fldin["vx1"];
  auto vx2 = fldin["vx2"];
  auto vx3 = fldin["vx3"];

  auto bx1 = fldin["bx1"];
  auto bx2 = fldin["bx2"];
  auto bx3 = fldin["bx3"];

  Shear &shear = this->shear;
  shear.Refresh(t);

  // Project the velocity field to be divergence free
  astra_for("mhd_projector", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));
      real k2t = kx1t*kx1t + kx2t*kx2t + kx3t*kx3t;
      if(k2t > 0.0) {
        complex kv_dot_v = kx1t*vx1(i,j,k)+kx2t*vx2(i,j,k)+kx3t*vx3(i,j,k);
        vx1(i,j,k) -= kv_dot_v*kx1t/k2t;
        vx2(i,j,k) -= kv_dot_v*kx2t/k2t;
        vx3(i,j,k) -= kv_dot_v*kx3t/k2t;

        complex kv_dot_b = kx1t*bx1(i,j,k)+kx2t*bx2(i,j,k)+kx3t*bx3(i,j,k);
        bx1(i,j,k) -= kv_dot_b*kx1t/k2t;
        bx2(i,j,k) -= kv_dot_b*kx2t/k2t;
        bx3(i,j,k) -= kv_dot_b*kx3t/k2t;
      }
  });
  astra::popRegion();
}

template <typename Shear>
void Mhd<Shear>::ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) {
  astra::pushRegion("Mhd::ImplicitStep");
  Projector(fldin, t);
  auto kx1 = this->grid->kx[IDIR];
  auto kx2 = this->grid->kx[JDIR];
  auto kx3 = this->grid->kx[KDIR];

  auto vx1 = fldin["vx1"];
  auto vx2 = fldin["vx2"];
  auto vx3 = fldin["vx3"];
  auto bx1 = fldin["bx1"];
  auto bx2 = fldin["bx2"];
  auto bx3 = fldin["bx3"];

  const real nu= this->nu;
  const real eta = this->eta;
  const int n = this->viscosityOrder;
  const int p = this->resistivityOrder;
  Shear shear = this->shear;
  shear.Refresh(t);
  astra_for("mhd_diffusion", fldin,
    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
      const real kx1t = shear.kx1t(kx1(i),kx2(j),kx3(k));
      const real kx2t = shear.kx2t(kx1(i),kx2(j),kx3(k));
      const real kx3t = shear.kx3t(kx1(i),kx2(j),kx3(k));
      const real k2t = kx1t*kx1t + kx2t*kx2t + kx3t*kx3t;

      //real factor = (1-0.5*dt*nu*k2)/(1+0.5*dt*nu*k2); // Crank-Nicholson
      real factor_nu = std::exp(-dt * pow(nu*k2t, n)); // Exact integration
      real factor_eta = std::exp(-dt * pow(eta*k2t, p)); // Exact integration
      vx1(i,j,k) *= factor_nu;
      vx2(i,j,k) *= factor_nu;
      vx3(i,j,k) *= factor_nu;
      bx1(i,j,k) *= factor_eta;
      bx2(i,j,k) *= factor_eta;
      bx3(i,j,k) *= factor_eta;
    });
  astra::popRegion();
}

template <typename Shear>
real Mhd<Shear>::GetInvDt() {
  astra::pushRegion("Mhd::GetInvDt");
  real invdt = 0.0;
  real kx1max = this->shear.kx1tmax();
  real kx2max = this->shear.kx2tmax();
  real kx3max = this->shear.kx3tmax();

  auto vr1 = this->vr1;
  auto vr2 = this->vr2;
  auto vr3 = this->vr3;

  auto br1 = this->br1;
  auto br2 = this->br2;
  auto br3 = this->br3;

  real Omega = std::fabs(this->Omega);
  real S = std::fabs(this->shear.shearRate);
  astra_reduce("mhd_timestep_reduction",
    0,npr[IDIR],
    0,npr[JDIR],
    0,npr[KDIR],
    KOKKOS_LAMBDA (int64_t i, int64_t j, int64_t k, real &dtmax) {
      real idtx1 = std::fabs(vr1(i,j,k)*kx1max);
      real idtx2 = std::fabs(vr2(i,j,k)*kx2max);
      real idtx3 = std::fabs(vr3(i,j,k)*kx3max);

      real gamma_v = idtx1+idtx2+idtx3;
      gamma_v += Omega; // rotation source term
      gamma_v += S; // shear source term

      idtx1 = std::fabs(br1(i,j,k)*kx1max); // Alfvén velocity contribution
      idtx2 = std::fabs(br2(i,j,k)*kx2max);
      idtx3 = std::fabs(br3(i,j,k)*kx3max);

      real gamma_b = idtx1+idtx2+idtx3;
      dtmax = std::fmax(dtmax,gamma_v+gamma_b);
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
std::vector<std::string> Mhd<Shear>::GetVariables() {
  return {"vx1","vx2","vx3","bx1","bx2","bx3"};
}

template <typename Shear>
void Mhd<Shear>::PostStage(Field<Array3D<complex>>& fldin, real t) {
  if constexpr(Shear::isEnabled) {
    if(this->shear.NeedRemap(t)) {
      this->shear.Remap(t, fldin["vx1"]);
      this->shear.Remap(t, fldin["vx2"]);
      this->shear.Remap(t, fldin["vx3"]);
      this->shear.Remap(t, fldin["bx1"]);
      this->shear.Remap(t, fldin["bx2"]);
      this->shear.Remap(t, fldin["bx3"]);

      this->shear.SetTinit(t);
    }
  }
  this->Projector(fldin, t);
}

#endif // RIGHTHANDSIDE_MHD_HPP_
