// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <Kokkos_Core.hpp>
#include "initFlow.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "arrays.hpp"
#include "field.hpp"
#include "global.hpp"
#include "loop.hpp"
#include "fft.hpp"

InitFlow::InitFlow(Input &input, Grid *grid) : grid(grid), input(&input) {
  for(int dir = 0 ; dir < 3 ; dir++) {
    kx[dir] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),grid->kx[dir]);
    x[dir] =  Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),grid->x[dir]);
  }
};


void InitFlow::Init(Field<Array3D<complex>>& field) {
  // Initialize the flow according to input parameters
  // Make a host copy of the field
  Field<ArrayHost3D<complex>> hfield("hfield", field);
  hfield.Reset();

  if(input->CheckEntry("InitFlow","shear_layer")>=0) {
    ShearLayer(hfield);
  }
  if(input->CheckEntry("InitFlow","mean_field")>=0) {
    MeanField(hfield);
  }
  if(input->CheckEntry("InitFlow","large_scale_noise")>=0) {
    LargeScale3DNoise(hfield);
  }
  if(input->CheckEntry("InitFlow","large_scale_2d_noise")>=0) {
    LargeScale2DNoise(hfield);
  }
  if(input->CheckEntry("InitFlow","large_scale_1d_noise")>=0) {
    LargeScale1DNoise(hfield);
  }

  Projector(hfield);
  // Copy back to device
  field.CopyFrom(hfield);
}

void InitFlow::ShearLayer(Field<ArrayHost3D<complex>>& hfieldOut) {
  // Implementation of mean field initialization
  ArrayHost1D<real> x1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),grid->x[IDIR]);
  ArrayHost1D<real> x2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),grid->x[JDIR]);
  ArrayHost1D<real> x3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),grid->x[KDIR]);
  real y0 = input->Get<real>("InitFlow","shear_layer",0);
  real v0 = input->Get<real>("InitFlow","shear_layer",1);

  Field<ArrayHost3D<real>> realField("realField", grid->npr, hfieldOut);
  Field<ArrayHost3D<complex>> complexField("complexField", grid->npf, hfieldOut);
  realField.Reset();
  complexField.Reset();

  for(int i = 0 ; i < grid->npr_glob[IDIR] ; i++) {
    for(int j = 0 ; j < grid->npr_glob[JDIR] ; j++) {
      real y = x2(j);
      for(int k = 0 ; k < grid->npr_glob[KDIR] ; k++) {
        if(std::fabs(y) < y0) {
          realField["vx1"](i,j,k) = -v0;
        } else {
          realField["vx1"](i,j,k) = v0;
        }
      }
    }
  }

  // Fourier transform to get the complex field
  grid->fft->R2C_Host(realField["vx1"], complexField["vx1"]);
  
  // add to output field
  for(auto it : hfieldOut) {
    auto viewOut = it.second;
    auto viewIn = complexField[it.first];
    for(int i = 0 ; i < grid->npf[IDIR] ; i++) {
      for(int j = 0 ; j < grid->npf[JDIR] ; j++) {
        for(int k = 0 ; k < grid->npf[KDIR] ; k++) {
          viewOut(i,j,k) += viewIn(i,j,k);
        }
      }
    }
  }
}

void InitFlow::MeanField(Field<ArrayHost3D<complex>>& hfield) {
  // Implementation of mean field initialization
  int64_t ntot = grid->npr_glob[IDIR]*grid->npr_glob[JDIR]*grid->npr_glob[KDIR];
  hfield["vx1"](0,0,0) = input->Get<real>("InitFlow","mean_field",0)*ntot;
  hfield["vx2"](0,0,0) = input->Get<real>("InitFlow","mean_field",1)*ntot;
  hfield["vx3"](0,0,0) = input->Get<real>("InitFlow","mean_field",2)*ntot;
}



void InitFlow::LargeScale3DNoise(Field<ArrayHost3D<complex>>& field) {
  // Implementation of large scale noise initialization
  int64_t ntot = grid->npr_glob[IDIR]*grid->npr_glob[JDIR]*grid->npr_glob[KDIR];
  real noiseAmplitude = input->Get<real>("InitFlow","large_scale_noise",0);
  real noiseCutLength = input->Get<real>("InitFlow","large_scale_noise",1);

  real lx = grid->xend_glob[IDIR]-grid->xbeg_glob[IDIR];
  real ly = grid->xend_glob[JDIR]-grid->xbeg_glob[JDIR];
  real lz = grid->xend_glob[KDIR]-grid->xbeg_glob[KDIR];
  // Number of modes that are excited (approx)
  real nmodes = lx*ly*lz/(noiseCutLength*noiseCutLength*noiseCutLength);

  for(int k = 0 ; k < grid->npf[KDIR] ; k++) {
    for(int j = 0 ; j < grid->npf[JDIR] ; j++) {
      for(int i = 0 ; i < grid->npf[IDIR] ; i++) {
        real ktot = std::sqrt(kx[IDIR](i)*kx[IDIR](i)+
                              kx[JDIR](j)*kx[JDIR](j)+
                              kx[KDIR](k)*kx[KDIR](k))
                                /(2.0*M_PI);
        if(ktot*noiseCutLength < 1.0) {
          for(auto& it : field) {
            auto view = it.second;
            real phase = 2.0*M_PI*astra::randm();
            real ampl = noiseAmplitude;
            view(i,j,k) += Kokkos::complex(ampl*std::cos(phase), ampl*std::sin(phase))/nmodes*ntot;
          }
        }
      }
    }
  } 
}

void InitFlow::LargeScale2DNoise(Field<ArrayHost3D<complex>>& field) {
  // Implementation of large scale noise initialization
  int64_t ntot = grid->npr_glob[IDIR]*grid->npr_glob[JDIR]*grid->npr_glob[KDIR];
  real noiseAmplitude = input->Get<real>("InitFlow","large_scale_2d_noise",0);
  real noiseCutLength = input->Get<real>("InitFlow","large_scale_2d_noise",1);

  real lx = grid->xend_glob[IDIR]-grid->xbeg_glob[IDIR];
  real ly = grid->xend_glob[JDIR]-grid->xbeg_glob[JDIR];

  // Number of modes that are excited (approx)
  real nmodes = lx*ly/(noiseCutLength*noiseCutLength);

  for(int k = 0 ; k < grid->npf[KDIR] ; k++) {
    for(int j = 0 ; j < grid->npf[JDIR] ; j++) {
      for(int i = 0 ; i < grid->npf[IDIR] ; i++) {
        real ktot = std::sqrt(kx[IDIR](i)*kx[IDIR](i)+
                              kx[JDIR](j)*kx[JDIR](j))
                                /(2.0*M_PI);
        if(ktot*noiseCutLength < 1.0 && kx[KDIR](k) == 0.0) {
          for(auto& it : field) {
            auto view = it.second;
            real phase = 2.0*M_PI*astra::randm();
            real ampl = noiseAmplitude;
            view(i,j,k) += Kokkos::complex(ampl*std::cos(phase), ampl*std::sin(phase))/nmodes*ntot;
          }
        }
      }
    }
  } 
}

void InitFlow::LargeScale1DNoise(Field<ArrayHost3D<complex>>& field) {
  // Implementation of large scale noise initialization
  int64_t ntot = grid->npr_glob[IDIR]*grid->npr_glob[JDIR]*grid->npr_glob[KDIR];
  real noiseAmplitude = input->Get<real>("InitFlow","large_scale_1d_noise",0);
  real noiseCutLength = input->Get<real>("InitFlow","large_scale_1d_noise",1);

  real lx = grid->xend_glob[IDIR]-grid->xbeg_glob[IDIR];

  // Number of modes that are excited (approx)
  real nmodes = lx/(noiseCutLength);

  for(int k = 0 ; k < grid->npf[KDIR] ; k++) {
    for(int j = 0 ; j < grid->npf[JDIR] ; j++) {
      for(int i = 0 ; i < grid->npf[IDIR] ; i++) {
        real ktot = std::sqrt(kx[IDIR](i)*kx[IDIR](i)+
                              kx[JDIR](j)*kx[JDIR](j))
                                /(2.0*M_PI);
        if(ktot*noiseCutLength < 1.0 && kx[KDIR](k) == 0.0 && kx[JDIR](j) == 0.0) {
          for(auto& it : field) {
            auto view = it.second;
            real phase = 2.0*M_PI*astra::randm();
            real ampl = noiseAmplitude;
            view(i,j,k) += Kokkos::complex(ampl*std::cos(phase), ampl*std::sin(phase))/nmodes*ntot;
          }
        }
      }
    }
  } 
}

void InitFlow::Projector(Field<ArrayHost3D<complex>>& fldin) {
  auto kx1 = this->kx[IDIR];
  auto kx2 = this->kx[JDIR];
  auto kx3 = this->kx[KDIR];

  try {
    auto vx1 = fldin["vx1"];
    auto vx2 = fldin["vx2"];
    auto vx3 = fldin["vx3"];

    // Project the velocity field to be divergence free
    for(int k = 0 ; k < grid->npf[KDIR] ; k++) {
      for(int j = 0 ; j < grid->npf[JDIR] ; j++) {
        for(int i = 0 ; i < grid->npf[IDIR] ; i++) {
        real k2 = kx1(i)*kx1(i)+kx2(j)*kx2(j)+kx3(k)*kx3(k);
        if(k2 > 0.0) {
          complex kv_dot_v = kx1(i)*vx1(i,j,k)+kx2(j)*vx2(i,j,k)+kx3(k)*vx3(i,j,k);
          vx1(i,j,k) -= kv_dot_v*kx1(i)/k2;
          vx2(i,j,k) -= kv_dot_v*kx2(j)/k2;
          vx3(i,j,k) -= kv_dot_v*kx3(k)/k2;
        }
    }}}  
  } catch(...) {
    astra::cerr << "Projector: can't apply projection on this field." << std::endl;  
  }
}