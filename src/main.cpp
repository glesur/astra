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
  Kokkos::initialize( argc, argv );
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

    // Init the right hand sides
    auto rhsVector = RightHandSideFactory<Array3D<complex>>::Create(input, &grid);
    // Init the time integrator
    auto *timeIntegrator = TimeIntegratorFactory<Array3D<complex>>::Create(input, &grid, rhsVector);

    // Create a state matching rhsVector requirements
    Field<Array3D<complex>> state("state",grid.npf);
    for(auto rhs : rhsVector) {
      for(auto var : rhs->GetVariables()) {
        state.Add(var);
      }
    }

    // Initial conditions
    initFlow.Init(state);

    // Create a time integrator
    //TimeIntegrator<Array3D<complex>> *timeIntegrator = new RK3TimeIntegrator<Array3D<complex>>(input, &grid, rhsVector);
    
    int nvtk = 0;
    // Test the time integrator
    
    Kokkos::Timer timer;

    while(timeIntegrator->GetCycle() < 100) {
      if(timeIntegrator->GetCycle()%10==0) {
        WriteFile(input, &grid, state, nvtk, timeIntegrator->GetTime());
        nvtk++;
      }
      timeIntegrator->Cycle(state);
    }
    WriteFile(input, &grid, state, nvtk, timeIntegrator->GetTime());

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
  return(0);
}


