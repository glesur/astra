// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************


#include "field.hpp"
#include "input.hpp"
#include "global.hpp"
#include "reduce.hpp"
#include "grid.hpp"
#include "loop.hpp"
#include "timevarSpectrum.hpp"
#include "linearshear.hpp"
#include <iostream>
#include <fstream>
#ifdef WITH_MPI
#include <mpi.h>
#endif

TimeVarSpectrum::TimeVarSpectrum(Input &input, Grid *grid, std::string name, std::string directory) : TimeVar(input, grid, name, directory) {
  const std::string token = "spectrum_";
  size_t tokenLocation = this->name.find(token);
  if(tokenLocation == std::string::npos) {
    throw std::runtime_error("TimeVarSpectrum: variable name must start with 'spectrum_'");
  }
  std::string varNames = this->name.substr(tokenLocation + token.length());
  
  size_t dotLocation = varNames.find(".");
  var1 = varNames.substr(0, dotLocation);
  var2 = varNames.substr(dotLocation+1);

  // translate variable names from snoopy
  var1 = this->ExtractVarName(var1);
  var2 = this->ExtractVarName(var2);

  // Compute spectral bins
  real lmax = 0;
  this->kmax = 0;

  for(int dir = 0 ; dir < 3 ; dir++) {
    lmax = std::fmax(lmax,grid->xend_glob[dir]-grid->xbeg_glob[dir]);
    kmax = std::fmax(kmax,grid->kmax[dir]);
  }
  this->kmin = 2.0*M_PI/lmax;
  this->nbins = static_cast<int>(kmax/kmin);
  if(this->nbins < 1) this->nbins = 1;

  // Check if we have shear
  std::string shearTypeStr = input.GetOrSet<std::string>("Physics","shear_type",0,"disabled");
  if(shearTypeStr =="linear") {
    this->haveLinearShear = true;
    this->linearShear = std::make_unique<LinearShear>(input, grid);
  }

  // Write bins file
  fs::path kfile = this->filename;
  kfile.replace_filename(this->filename.stem().string()+"_k.dat");
  if(this->isRoot) {
    std::ofstream outfile;
    outfile.open(kfile, std::ios_base::trunc);
    outfile.precision(10);
    for(int i = 0; i < this->nbins; i++) {
      real kvalue = (i + 0.5)*this->kmin;
      outfile << std::scientific << kvalue << "\t";
    }
    outfile << std::endl;
    outfile.close();
  }
};

void TimeVarSpectrum::Write(const real t, Field<Array3D<complex>>& field, Field<Array3D<real>>& fieldReal) {
    astra::pushRegion("TimeVarSpectrum::Write");
    // Compute energy

    auto view1 = field[var1];
    auto view2 = field[var2];

    real kmax = this->kmax;
    int nbins = this->nbins;
    real kmin = this->kmin;
    
    Array1D<real> spectrum("spectrum", nbins);
    Kokkos::deep_copy(spectrum, 0.0);

    auto kx1 = grid->kx[IDIR];
    auto kx2 = grid->kx[JDIR];
    auto kx3 = grid->kx[KDIR];

    LinearShear shear;

    bool haveLinearShear = this->haveLinearShear;
    if(this->haveLinearShear) {
      shear = *(this->linearShear.get());
      shear.SetTinit(t);
    }

    real npoints = static_cast<real>(grid->npr_glob[IDIR] * grid->npr_glob[JDIR] * grid->npr_glob[KDIR]);
    astra_for("timevar_spectrum_compute_"+name, field,
        KOKKOS_LAMBDA(int i,int j,int k) {
          real kxt,kyt,kzt;
          real fact= 2.0;
          if(kx3(k) == 0) fact = 1.0; // account for Hermitian symmetry

          fact=fact/(npoints*npoints);
          if(haveLinearShear) {
            kxt = shear.kx1t(kx1(i), kx2(j), kx3(k));
            kyt = shear.kx2t(kx1(i), kx2(j), kx3(k));
            kzt = shear.kx3t(kx1(i), kx2(j), kx3(k));
          } else {
            kxt = kx1(i);
            kyt = kx2(j);
            kzt = kx3(k);
          }
          real kw = std::sqrt(kxt*kxt + kyt*kyt + kzt*kzt);
          int bin = static_cast<int>(kw/kmin+0.5);
          if(bin >=0 && bin < nbins) {
            Kokkos::atomic_add(&spectrum(bin), fact * (view1(i,j,k) * Kokkos::conj(view2(i,j,k))).real());
          }
      });
    ArrayHost1D<real> spectrumHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), spectrum);
    #ifdef WITH_MPI
      ArrayHost1D<real> spectrumHostGlob = Kokkos::create_mirror(Kokkos::HostSpace(), spectrum);
      // Reduce across all processes
      real q0;
      MPI_Reduce(spectrumHost.data(), spectrumHostGlob.data(), nbins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      spectrumHost = spectrumHostGlob;
    #endif
    // Write to file
    if(this->isRoot) {
      std::ofstream outfile;
      outfile.open(this->filename, std::ios_base::app);
      outfile.precision(10);
      outfile << std::scientific << t << "\t";
      for(int i = 0; i < nbins; i++) {
        outfile << spectrumHost(i) << (i == nbins - 1 ? "\n" : "\t");
      }
      outfile.close();
    }
    astra::popRegion();
  }