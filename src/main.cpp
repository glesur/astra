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
#include "dumpVariables.hpp"
#include "output.hpp"
#include "transpose.hpp"
#include "fft.hpp"
#include "checkNan.hpp"


void LogFinished(Grid &grid, Input &Tint, Kokkos::Timer &timer, TimeIntegrator<Array3D<complex>> *timeIntegrator);



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
    for(auto &rhs : rhsVector) {
      for(auto var : rhs->GetVariables()) {
        state.Add(var);
      }
    }

    // Init the time integrator
    auto timeIntegrator = TimeIntegratorFactory<Array3D<complex>>::Create(input, &grid, rhsVector);

    // Declare the state for I/O routines
    DumpVariables::Register("state", state);
    
    // Init the output.
    Output output(input, grid);

    /*********************************************************************************** */
    // From this point the basic logic is initialized. We needd to fill the state array
    // either from initial conditions or from a dump file, and then we can start 
    // the time integration loop.

    if(input.restartRequested) {
      output.RestartFromDump(input);
      timeIntegrator->Reset();  // Reset time proxies in the rhs after restart
    } else {
      ///////////////////////////////
      // Initial condition generation
      //////////////////////////////

      initFlow.Init(state);
      // Write initial condition
      output.CheckForOutput(input, state, timeIntegrator->GetTime());
    }


    astra::cout << "Main: Starting time integration..." << std::endl;
    
    real tstop = input.Get<real>("TimeIntegrator","tstop",0);
    Kokkos::Timer timer;

    /*************************************************************************** */
    // MAIN LOOP STARTS HERE
    /*************************************************************************** */
    while(timeIntegrator->GetTime() < tstop) {
      timeIntegrator->SetDtMax(tstop - timeIntegrator->GetTime());
      timeIntegrator->Cycle(state);

      // Output
      output.CheckForOutput(input, state, timeIntegrator->GetTime());

      // Abort checks
      bool abortRequested = false;  
      if(input.CheckForAbort()) {
        abortRequested = true;
        astra::cout << "Main: Abort signal received." << std::endl;
      }
      abortRequested = abortRequested || timeIntegrator->CheckForMaxRuntime();

      if(input.maxCycles>0 && timeIntegrator->GetCycle()>=input.maxCycles) {
        abortRequested = true;
        astra::cout << "Main: Maximum number of cycles reached." << std::endl;
      }

      if(abortRequested) {
        output.ForceDump(timeIntegrator->GetTime());
        astra::cout << "Main: Aborting time integration..." << std::endl;
        break;
      }
    }


    LogFinished(grid, input, timer, timeIntegrator.get());
    // Show profiler output
    astra::prof.Show();

    astra::cout << "Main: Job completed successfully." << std::endl;
  }
  Kokkos::finalize();

  #ifdef WITH_MPI
    MPI_Finalize();
  #endif

  return(0);
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

