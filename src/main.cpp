#include <stdlib.h>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

#include "astra.hpp"
#include "profiler.hpp"
#include "loop.hpp"
#include "global.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "initFlow.hpp"
#include "logger.hpp"
#include "field.hpp"
#include "timeIntegrator.hpp"
#include "timeIntegratorFactory.hpp"
#include "rightHandSideFactory.hpp"
#include "vtk.hpp"


void WriteFile(Input &input, 
               Grid *grid,
               Field<Array3D<complex>>& state,  
               int n,
               real time) {
  std::stringstream ssfileName, ssvtkFileNum;
  ssvtkFileNum << std::setfill('0') << std::setw(4) << n;
  std::string filename = std::string("data.")+ssvtkFileNum.str();
  Vtk vtk(input, grid, time, filename);
  vtk.Write(state);
}

void LogFinished(Grid &grid, Input &Tint, Kokkos::Timer &timer, TimeIntegrator<Array3D<complex>> *timeIntegrator) {
  int n_days{0}, n_hours{0}, n_minutes{0}, n_seconds{0};
  div_t divres;
  divres = div(timer.seconds(), 86400);
  n_days = divres.quot;
  divres = div(divres.rem, 3600);
  n_hours = divres.quot;
  divres = div(divres.rem, 60);
  n_minutes = divres.quot;
  n_seconds = divres.rem;

  astra::cout << "Main: Reached t=" << timeIntegrator->GetTime() << std::endl;
  double perfs = timer.seconds() / grid.npr_glob[IDIR] / grid.npr_glob[JDIR]
                    / grid.npr_glob[KDIR] / timeIntegrator->GetCycle() * astra::psize;
  astra::cout << "Main: Completed in ";
    if (n_days > 0) {
      astra::cout << n_days << " day";
      if (n_days != 1) {
        astra::cout << "s";
      }
      astra::cout << " ";
    }
    if (n_hours > 0) {
      astra::cout << n_hours << " hour";
      if (n_hours != 1) {
        astra::cout << "s";
      }
      astra::cout << " ";
    }
    if (n_minutes > 0) {
      astra::cout << n_minutes << " minute";
      if (n_minutes != 1) {
        astra::cout << "s";
      }
      astra::cout << " ";
    }
    astra::cout << n_seconds << " second";
    if (n_seconds != 1) {
      astra::cout << "s";
    }
    astra::cout << " ";
    astra::cout << "and " << timeIntegrator->GetCycle() << " cycle";
    if (timeIntegrator->GetCycle() != 1) {
      astra::cout << "s";
    }
    astra::cout << std::endl;
    astra::cout << "Main: ";
    astra::cout << "Perfs are " << std::scientific << 1/perfs << " cell updates/second" << std::endl;
}

int main( int argc, char* argv[] ) {
    bool initKokkosBeforeMPI = false;

  // When running on GPUS with Omnipath network,
  // Kokkos needs to be initialised *before* the MPI layer
#ifdef KOKKOS_ENABLE_CUDA
  if(std::getenv("PSM2_CUDA") != NULL) {
    initKokkosBeforeMPI = true;
  }
#endif

  if(initKokkosBeforeMPI)  Kokkos::initialize( argc, argv );

#ifdef WITH_MPI
  MPI_Init(&argc,&argv);
#endif

  if(!initKokkosBeforeMPI) Kokkos::initialize( argc, argv );
  {
    // Initialize astra screen outputs, logs, profiling
    astra::initialize();

    ///////////////////////////////////
    //Initialisation phase
    // Read input
    Input input(argc, argv);
    input.PrintLogo();

    Grid grid(input);
    InitFlow initFlow(input, &grid);
    
    
    // Show configuration after initialisation
    input.ShowConfig();
    grid.ShowConfig();

    #ifdef WITH_MPI
    grid.fft->TestTranspose();
    #endif

    // Init the right hand sides
    auto rhsVector = RightHandSideFactory<Array3D<complex>>::Create(input, &grid);

    // Create a state matching rhsVector requirements
    Field<Array3D<complex>> state("state",grid.npf);
    for(auto rhs : rhsVector) {
      for(auto var : rhs->GetVariables()) {
        state.Add(var);
      }
    }

    // Initial conditions
    initFlow.Init(state);

    // Write initial condition@
    real lastOutput = 0.0;
    real outputStep = input.GetOrSet<real>("Output","vtk",0,0.1);
    int nvtk = 0;
    WriteFile(input, &grid, state, nvtk, 0.0);
    nvtk++;

    astra::cout << "Main: Starting time integration..." << std::endl;
    // Init the time integrator
    auto *timeIntegrator = TimeIntegratorFactory<Array3D<complex>>::Create(input, &grid, rhsVector);
    real tstop = input.Get<real>("TimeIntegrator","tstop",0);
    Kokkos::Timer timer;
    while(timeIntegrator->GetTime() < tstop) {
      timeIntegrator->Cycle(state);
      if(timeIntegrator->GetTime()-lastOutput>=outputStep) {
        WriteFile(input, &grid, state, nvtk, timeIntegrator->GetTime());
        nvtk++;
        lastOutput += outputStep;
      }
    }

    LogFinished(grid, input, timer, timeIntegrator);
    // Show profiler output
    astra::prof.Show();

    // Clean up
    delete timeIntegrator;
    for(auto rhs : rhsVector) {
      delete rhs;
    }
    astra::cout << "Main: Job completed successfully." << std::endl;
  }
  Kokkos::finalize();

  #ifdef WITH_MPI
    MPI_Finalize();
  #endif

  return(0);
}


