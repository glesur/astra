// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include "hydro.hpp"
#include "arrays.hpp"
#include "grid.hpp"
#include "loop.hpp"
#include "reduce.hpp"
#include "global.hpp"
#include "fft.hpp"

Hydro::Hydro(Input &input, Grid *grid) : RightHandSide<Array3D<complex>>(input, grid) {
  // Allocate all of the temporary arrays
  vr1 = Array3D<real>("Hydro::vr1", grid->npr);
  vr2 = Array3D<real>("Hydro::vr2", grid->npr);
  vr3 = Array3D<real>("Hydro::vr3", grid->npr);
  wr11 = Array3D<real>("Hydro::wr11", grid->npr);
  wr12 = Array3D<real>("Hydro::wr12", grid->npr);
  wr13 = Array3D<real>("Hydro::wr13", grid->npr);
  wr22 = Array3D<real>("Hydro::wr22", grid->npr);
  wr23 = Array3D<real>("Hydro::wr23", grid->npr);
  wr33 = Array3D<real>("Hydro::wr33", grid->npr);
  wf11 = Array3D<complex>("Hydro::wf11", grid->npf);
  wf12 = Array3D<complex>("Hydro::wf12", grid->npf);
  wf13 = Array3D<complex>("Hydro::wf13", grid->npf);
  wf22 = Array3D<complex>("Hydro::wf22", grid->npf);
  wf23 = Array3D<complex>("Hydro::wf23", grid->npf);
  wf33 = Array3D<complex>("Hydro::wf33", grid->npf);
  npr=grid->npr;
  npf=grid->npf;

  this->nu = input.GetOrSet<real>("Hydro","viscosity",0,1e-3);
  astra::cout << "Hydro rhs with viscosity nu=" << nu << std::endl;
}

Hydro::~Hydro() {
}

void Hydro::ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) {
  astra::pushRegion("Hydro::ExplicitStep");
  // Fourier transform the velocity field to real space
  grid->fft->C2R(fldin["vx1"], vr1);
  grid->fft->C2R(fldin["vx2"], vr2);
  grid->fft->C2R(fldin["vx3"], vr3);

  // Compute the cross-correlation
  auto vr1 = this->vr1;
  auto vr2 = this->vr2;
  auto vr3 = this->vr3;
  auto wr11 = this->wr11;
  auto wr12 = this->wr12;
  auto wr13 = this->wr13;
  auto wr22 = this->wr22;
  auto wr23 = this->wr23;
  auto wr33 = this->wr33;
  astra_for("hydro_windup", 0,npr[IDIR],0,npr[JDIR],0,npr[KDIR],
    KOKKOS_LAMBDA(int i, int j, int k) {
      real v1 = vr1(i, j, k);
      real v2 = vr2(i, j, k);
      real v3 = vr3(i, j, k);
      wr11(i, j, k) = v1 * v1;
      wr12(i, j, k) = v1 * v2;
      wr13(i, j, k) = v1 * v3;
      wr22(i, j, k) = v2 * v2;
      wr23(i, j, k) = v2 * v3;
      wr33(i, j, k) = v3 * v3;
    });

  // Fourier transform back to spectral space
  grid->fft->R2C(wr11, wf11);
  grid->fft->R2C(wr12, wf12);
  grid->fft->R2C(wr13, wf13);
  grid->fft->R2C(wr22, wf22);
  grid->fft->R2C(wr23, wf23);
  grid->fft->R2C(wr33, wf33);
  
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

  real kx1max = grid->kmax[IDIR];
  real kx2max = grid->kmax[JDIR];
  real kx3max = grid->kmax[KDIR];
  // Compute the nonlinear term in spectral space
  astra_for("hydro_nonlinear", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int i, int j, int k) {
      // 2/3 de-aliasing rule
      complex mask = (std::fabs(kx1(i))< 2./3*kx1max 
                   && std::fabs(kx2(j))< 2./3*kx2max 
                   && std::fabs(kx3(k))< 2./3*kx3max) ? 1.0 : 0.0;
      dvx1(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1(i)*wf11(i,j,k)+kx2(j)*wf12(i,j,k)+kx3(k)*wf13(i,j,k))*mask;
      dvx2(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1(i)*wf12(i,j,k)+kx2(j)*wf22(i,j,k)+kx3(k)*wf23(i,j,k))*mask;
      dvx3(i,j,k) -= Kokkos::complex(0.0,1.0)*(kx1(i)*wf13(i,j,k)+kx2(j)*wf23(i,j,k)+kx3(k)*wf33(i,j,k))*mask;
  });

  Projector(dfld);
  astra::popRegion();
}

void Hydro::Projector(Field<Array3D<complex>>& fldin) {
  astra::pushRegion("Hydro::Projector");
  auto kx1 = this->grid->kx[IDIR];
  auto kx2 = this->grid->kx[JDIR];
  auto kx3 = this->grid->kx[KDIR];

  auto vx1 = fldin["vx1"];
  auto vx2 = fldin["vx2"];
  auto vx3 = fldin["vx3"];

  // Project the velocity field to be divergence free
  astra_for("hydro_projector", 0,npf[IDIR],0,npf[JDIR],0,npf[KDIR],
    KOKKOS_LAMBDA(int i, int j, int k) {
      real k2 = kx1(i)*kx1(i)+kx2(j)*kx2(j)+kx3(k)*kx3(k);
      if(k2 > 0.0) {
        complex kv_dot_v = kx1(i)*vx1(i,j,k)+kx2(j)*vx2(i,j,k)+kx3(k)*vx3(i,j,k);
        vx1(i,j,k) -= kv_dot_v*kx1(i)/k2;
        vx2(i,j,k) -= kv_dot_v*kx2(j)/k2;
        vx3(i,j,k) -= kv_dot_v*kx3(k)/k2;
      }
  });
  astra::popRegion();
}
void Hydro::ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) {
  astra::pushRegion("Hydro::ImplicitStep");
  Projector(fldin);
  auto kx1 = this->grid->kx[IDIR];
  auto kx2 = this->grid->kx[JDIR];
  auto kx3 = this->grid->kx[KDIR];

  auto vx1 = fldin["vx1"];
  auto vx2 = fldin["vx2"];
  auto vx3 = fldin["vx3"];

  real nu= this->nu;
  astra_for("hydro_viscosity", fldin,
    KOKKOS_LAMBDA(int i, int j, int k) {
      real k2 = kx1(i)*kx1(i)+kx2(j)*kx2(j)+kx3(k)*kx3(k);
      //real factor = (1-0.5*dt*nu*k2)/(1+0.5*dt*nu*k2); // Crank-Nicholson
      real factor = std::exp(-dt * nu*k2); // Exact integration
      vx1(i,j,k) *= factor;
      vx2(i,j,k) *= factor;
      vx3(i,j,k) *= factor;
    });
  astra::popRegion();
}

real Hydro::GetInvDt() {
  astra::pushRegion("Hydro::GetInvDt");
  real invdt = 0.0;
  real kx1max = grid->kmax[IDIR];
  real kx2max = grid->kmax[JDIR];
  real kx3max = grid->kmax[KDIR];

  auto vr1 = this->vr1;
  auto vr2 = this->vr2;
  auto vr3 = this->vr3;

  astra_reduce("Timestep_reduction_dust",
    0,npr[IDIR],
    0,npr[JDIR],
    0,npr[KDIR],
    KOKKOS_LAMBDA (int i, int j, int k, real &dtmax) {
      real idtx1 = std::fabs(vr1(i,j,k)*kx1max);
      real idtx2 = std::fabs(vr2(i,j,k)*kx2max);
      real idtx3 = std::fabs(vr3(i,j,k)*kx3max);
      dtmax = std::fmax(dtmax,idtx1+idtx2+idtx3);

        },
    Kokkos::Max<real>(invdt));

  #ifdef WITH_MPI
    // Across all MPI processes
    MPI_Allreduce(MPI_IN_PLACE, &invdt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  #endif
  astra::popRegion();
  return invdt;
}

std::vector<std::string> Hydro::GetVariables() {
  return {"vx1","vx2","vx3"};
}
