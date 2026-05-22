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
#include "checkNan.hpp"

InitFlow::InitFlow(Input &input, Grid *grid) : grid(grid), input(&input)
#ifdef WITH_PYTHON
  , astraPy(input)
#endif
{
  for(int dir = 0 ; dir < 3 ; dir++) {
    kx[dir] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),grid->kx[dir]);
    x[dir] =  Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),grid->x[dir]);
  }
}

void InitFlow::Init(Field<Array3D<complex>>& field) {
  // Initialize the flow according to input parameters
  // Make a host copy of the field
  Field<ArrayHost3D<complex>> hfield("hfield", field);
  hfield.Reset();
  if(input->CheckEntry("InitFlow","python")>=0) {
    #ifdef WITH_PYTHON
      astraPy.InitFlow(grid, hfield);
    #else
      throw std::runtime_error("Python initial conditions were required, but Python support is disabled \
                                in this build of ASTRA. \n Please recompile with WITH_PYTHON=ON.");
    #endif
  }
  if(input->CheckEntry("InitFlow","shear_layer")>=0) {
    ShearLayer(hfield);
  }
  if(input->CheckEntry("InitFlow","mean_flow")>=0) {
    MeanFlow(hfield);
  }
  if(input->CheckEntry("InitFlow","mean_field")>=0) {
    MeanField(hfield);
  }
  if(input->CheckEntry("InitFlow","large_scale_3d_noise")>=0) {
    LargeScale3DNoise(hfield);
  }
  if(input->CheckEntry("InitFlow","large_scale_2d_noise")>=0) {
    LargeScale2DNoise(hfield);
  }
  if(input->CheckEntry("InitFlow","large_scale_1d_noise")>=0) {
    LargeScale1DNoise(hfield);
  }
  if(input->CheckEntry("InitFlow","pure_sine")>=0) {
    PureSine(hfield);
  }

  Projector(hfield);
  // Copy back to device
  field.CopyFrom(hfield);
  astra::CheckNan(field);
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

  for(int64_t i = 0 ; i < grid->npr[IDIR] ; i++) {
    for(int64_t j = 0 ; j < grid->npr[JDIR] ; j++) {
      real y = x2(j);
      for(int64_t k = 0 ; k < grid->npr[KDIR] ; k++) {
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
    for(int64_t i = 0 ; i < grid->npf[IDIR] ; i++) {
      for(int64_t j = 0 ; j < grid->npf[JDIR] ; j++) {
        for(int64_t k = 0 ; k < grid->npf[KDIR] ; k++) {
          viewOut(i,j,k) += viewIn(i,j,k);
        }
      }
    }
  }
}

void InitFlow::MeanFlow(Field<ArrayHost3D<complex>>& hfield) {
  // Implementation of mean flow initialization
  int64_t ntot = grid->npr_glob[IDIR];
  ntot *= grid->npr_glob[JDIR];
  ntot *= grid->npr_glob[KDIR];
  if(astra::prank==0) {
    hfield["vx1"](0,0,0) = input->Get<real>("InitFlow","mean_flow",0)*ntot;
    hfield["vx2"](0,0,0) = input->Get<real>("InitFlow","mean_flow",1)*ntot;
    hfield["vx3"](0,0,0) = input->Get<real>("InitFlow","mean_flow",2)*ntot;
  }
}

void InitFlow::MeanField(Field<ArrayHost3D<complex>>& hfield) {
  // Implementation of mean field initialization
  int64_t ntot = grid->npr_glob[IDIR];
  ntot *= grid->npr_glob[JDIR];
  ntot *= grid->npr_glob[KDIR];
  if(astra::prank==0) {
    hfield["bx1"](0,0,0) = input->Get<real>("InitFlow","mean_field",0)*ntot;
    hfield["bx2"](0,0,0) = input->Get<real>("InitFlow","mean_field",1)*ntot;
    hfield["bx3"](0,0,0) = input->Get<real>("InitFlow","mean_field",2)*ntot;
  }
}



void InitFlow::LargeScale3DNoise(Field<ArrayHost3D<complex>>& field) {
  astra::cout << "Adding large scale 3D noise to the flow" << std::endl;
  // Implementation of large scale noise initialization
  int64_t ntot = grid->npr_glob[IDIR];
  ntot *= grid->npr_glob[JDIR];
  ntot *= grid->npr_glob[KDIR];
  real noiseAmplitude = input->Get<real>("InitFlow","large_scale_3d_noise",0);
  real noiseCutLength = input->Get<real>("InitFlow","large_scale_3d_noise",1);

  real lx = grid->xend_glob[IDIR]-grid->xbeg_glob[IDIR];
  real ly = grid->xend_glob[JDIR]-grid->xbeg_glob[JDIR];
  real lz = grid->xend_glob[KDIR]-grid->xbeg_glob[KDIR];
  // Number of modes that are excited (approx)
  // real nmodes = lx*ly*lz/(noiseCutLength*noiseCutLength*noiseCutLength);

  // Count the number of modes excited
  real nmodes = 0;
  for(int64_t k = 0 ; k < grid->npf[KDIR] ; k++) {
    for(int64_t j = 0 ; j < grid->npf[JDIR] ; j++) {
      for(int64_t i = 0 ; i < grid->npf[IDIR] ; i++) {
        real ktot = std::sqrt(kx[IDIR](i)*kx[IDIR](i)+
                              kx[JDIR](j)*kx[JDIR](j)+
                              kx[KDIR](k)*kx[KDIR](k))
                                /(2.0*M_PI);
        if(ktot*noiseCutLength < 1.0 && ktot>0.0) {
          nmodes += 1.0;
        }
      }
    }
  }
  #ifdef WITH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &nmodes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  #endif
  real fact = pow(nmodes, 0.5);
  for(int64_t k = 0 ; k < grid->npf[KDIR] ; k++) {
    for(int64_t j = 0 ; j < grid->npf[JDIR] ; j++) {
      for(int64_t i = 0 ; i < grid->npf[IDIR] ; i++) {
        real ktot = std::sqrt(kx[IDIR](i)*kx[IDIR](i)+
                              kx[JDIR](j)*kx[JDIR](j)+
                              kx[KDIR](k)*kx[KDIR](k))
                                /(2.0*M_PI);
        if(ktot*noiseCutLength < 1.0 && ktot>0.0) {
          for(auto& it : field) {
            auto view = it.second;
            const real ampl = noiseAmplitude*astra::randm();
            const real phase = 2.0*M_PI*astra::randm();

            view(i,j,k) += Kokkos::complex(ampl*std::cos(phase), ampl*std::sin(phase))/fact*ntot;
          }
        }
      }
    }
  }
}

void InitFlow::LargeScale2DNoise(Field<ArrayHost3D<complex>>& field) {
  // Implementation of large scale noise initialization
  int64_t ntot = grid->npr_glob[IDIR];
  ntot *= grid->npr_glob[JDIR];
  ntot *= grid->npr_glob[KDIR];
  real noiseAmplitude = input->Get<real>("InitFlow","large_scale_2d_noise",0);
  real noiseCutLength = input->Get<real>("InitFlow","large_scale_2d_noise",1);

  real lx = grid->xend_glob[IDIR]-grid->xbeg_glob[IDIR];
  real ly = grid->xend_glob[JDIR]-grid->xbeg_glob[JDIR];

  // Number of modes that are excited (approx)
  real nmodes = lx*ly/(noiseCutLength*noiseCutLength);

  for(int64_t k = 0 ; k < grid->npf[KDIR] ; k++) {
    for(int64_t j = 0 ; j < grid->npf[JDIR] ; j++) {
      for(int64_t i = 0 ; i < grid->npf[IDIR] ; i++) {
        real ktot = std::sqrt(kx[IDIR](i)*kx[IDIR](i)+
                              kx[JDIR](j)*kx[JDIR](j))
                                /(2.0*M_PI);
        if(ktot*noiseCutLength < 1.0 && kx[KDIR](k) == 0.0 && ktot>0.0) {
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
  int64_t ntot = grid->npr_glob[IDIR];
  ntot *= grid->npr_glob[JDIR];
  ntot *= grid->npr_glob[KDIR];
  real noiseAmplitude = input->Get<real>("InitFlow","large_scale_1d_noise",0);
  real noiseCutLength = input->Get<real>("InitFlow","large_scale_1d_noise",1);

  real lx = grid->xend_glob[IDIR]-grid->xbeg_glob[IDIR];

  // Number of modes that are excited (approx)
  real nmodes = lx/(noiseCutLength);

  for(int64_t k = 0 ; k < grid->npf[KDIR] ; k++) {
    for(int64_t j = 0 ; j < grid->npf[JDIR] ; j++) {
      for(int64_t i = 0 ; i < grid->npf[IDIR] ; i++) {
        real ktot = std::sqrt(kx[IDIR](i)*kx[IDIR](i)+
                              kx[JDIR](j)*kx[JDIR](j))
                                /(2.0*M_PI);
        if(ktot*noiseCutLength < 1.0 && kx[KDIR](k) == 0.0 && kx[JDIR](j) == 0.0 && ktot>0.0) {
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

void InitFlow::PureSine(Field<ArrayHost3D<complex>>& field) {
  // Implementation of pure sine initialization
  const int direction = input->Get<int>("InitFlow","pure_sine",0);
  const real amplitude = input->Get<real>("InitFlow","pure_sine",1);


  for(auto& it : field) {
    auto view = it.second;
    if(direction == IDIR) {
      view(1,0,0) += amplitude;
    } else if(direction == JDIR) {
      view(0,1,0) += amplitude;
    } else if(direction == KDIR) {
      view(0,0,1) += amplitude;
    } else {
      throw std::runtime_error("pure_sine direction must be 0, 1 or 2");
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
    for(int64_t k = 0 ; k < grid->npf[KDIR] ; k++) {
      for(int64_t j = 0 ; j < grid->npf[JDIR] ; j++) {
        for(int64_t i = 0 ; i < grid->npf[IDIR] ; i++) {
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
