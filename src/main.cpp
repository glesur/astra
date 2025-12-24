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
#include "timevar.hpp"
#include "transpose.hpp"
#include "fft.hpp"


void WriteFile(Input &input, 
               Grid *grid,
               Field<Array3D<complex>>& state,  
               int n,
               real time) {
  std::string outputDirectory;
  if(input.CheckEntry("Output","vtk_dir")>=0) {
    outputDirectory = input.Get<std::string>("Output","vtk_dir",0);
  } else {
    outputDirectory = "./";
  }
  std::stringstream ssfileName, ssvtkFileNum;
  ssvtkFileNum << std::setfill('0') << std::setw(4) << n;
  std::string filename = std::string("data.")+ssvtkFileNum.str();
  Vtk vtk(grid, time, filename,outputDirectory);
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
  double perfs = timer.seconds() / grid.npr[IDIR] / grid.npr[JDIR]
                    / grid.npr[KDIR] / timeIntegrator->GetCycle();
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
    /*
    #ifdef WITH_MPI
    Transpose<complex> transpose_c(grid.npf);
    transpose_c.Test();
    Transpose<real> transpose_r(grid.npr);
    transpose_r.Test();
    grid.fft->TestMPI();
    #endif
    */
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

    // Whether we want timevar
    TimeVarOutput timevarOutput(input, &grid);
    timevarOutput.Reset();
    timevarOutput.Write(state, 0.0);
    real lastTimevar = 0.0;
    real timevarStep = input.GetOrSet<real>("Output","timevar_step",0,-1.0);

    astra::cout << "Main: Starting time integration..." << std::endl;
    // Init the time integrator
    auto *timeIntegrator = TimeIntegratorFactory<Array3D<complex>>::Create(input, &grid, rhsVector);
    real tstop = input.Get<real>("TimeIntegrator","tstop",0);
    Kokkos::Timer timer;
    while(timeIntegrator->GetTime() < tstop) {
      timeIntegrator->SetDtMax(tstop - timeIntegrator->GetTime());
      timeIntegrator->Cycle(state);

      // Output
      if(timeIntegrator->GetTime()-lastOutput>=outputStep) {
        WriteFile(input, &grid, state, nvtk, timeIntegrator->GetTime());
        nvtk++;
        lastOutput += outputStep;
      }
      if(timevarStep>0.0 && timeIntegrator->GetTime()-lastTimevar>=timevarStep) {
        timevarOutput.Write(state, timeIntegrator->GetTime());
        lastTimevar += timevarStep;
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


