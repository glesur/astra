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
#include "dump.hpp"
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

std::string CreateDumpFileName(int n) {
  std::stringstream ssdmpFileNum;
  ssdmpFileNum << std::setfill('0') << std::setw(4) << n;
  return std::string("dump.")+ssdmpFileNum.str()+std::string(".dmp");
}

// Helper function to convert filesystem::file_time into std::time_t
// see https://stackoverflow.com/questions/56788745/
// This conversion "hack" is required in C++17 as no proper conversion bewteen
// fs::last_write_time and std::time_t
// exists in the standard library until C++20
template <typename TP>
std::time_t to_time_t(TP tp) {
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>
                (tp - TP::clock::now() + std::chrono::system_clock::now());
    return std::chrono::system_clock::to_time_t(sctp);
}

int GetLastDumpInDirectory(std::string directory_str) {
  int num = -1;

  fs::path directory = directory_str;
  std::time_t youngFileTime;
  bool first = true;
  for (const auto & entry : fs::directory_iterator(directory)) {
      // Check file extension
      if(entry.path().extension().string().compare(".dmp")==0) {
        auto fileTime = to_time_t(fs::last_write_time(entry.path()));
        // Check which one is the most recent
        if(first || fileTime>youngFileTime) {
          // std::tm *gmt = std::gmtime(&fileTime);
          // idfx::cout << "file " << entry.path() << "is the most recent with "
          //            << std::put_time(gmt, "%d %B %Y %H:%M:%S") << std::endl;

          // Ours is more recent, extract the dump file number
          try {
            num = std::stoi(entry.path().filename().string().substr(5,4));
            first = false;
            youngFileTime = fileTime;
          } catch (...) {
            // the file name does not follow the convention "filebase.xxxx.dmp"
            // ->We skip it
          }
        }
      }
  }
  return(num);
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
    for(auto &rhs : rhsVector) {
      for(auto var : rhs->GetVariables()) {
        state.Add(var);
      }
    }

    // Initialise output variables
    real t0 = 0.0;
    real lastVtkOutput = 0.0;
    real outputVtkStep = input.GetOrSet<real>("Output","vtk",0,0.1);
    int nvtk = 0;

    real lastDmpOutput = 0.0;
    real outputDmpStep = input.GetOrSet<real>("Output","dump",0,-1);
    std::string outputDmpDir = input.GetOrSet<std::string>("Output","dump_dir",0,"./");
    int ndmp = 0;

    // Whether we want timevar
    TimeVarOutput timevarOutput(input, &grid);
    real lastTimevar = 0.0;
    real timevarStep = input.GetOrSet<real>("Output","timevar_step",0,-1.0);

    if(input.restartRequested) {
      ///////////////////////////////
      // Restarting from dump file
      ///////////////////////////////
      int restartFileNumber = input.restartFileNumber;;
      if(restartFileNumber==-1) {
        // Find the last dump file in the output directory
        restartFileNumber = GetLastDumpInDirectory(outputDmpDir);
        if(restartFileNumber<0) {
          throw std::runtime_error("No dump file found in directory "+outputDmpDir+" for restart.");
        }
      }
      std::string filename;

      if(restartFileNumber==-2) {
        // Restart from Snoopy dump
        filename = input.Get<std::string>("CommandLine","snoopy_restart",0);
        astra::cout << "Main: Restarting from Snoopy dump file " << filename << std::endl;
      } else {
        filename = outputDmpDir+"/"+CreateDumpFileName(restartFileNumber);
        astra::cout << "Main: Restarting from dump file number " << restartFileNumber << std::endl;
      }
      
      Dump dump(&grid, filename);
      if(restartFileNumber==-2) {
        dump.ReadSnoopy();
      } else {
        dump.Read();
      }
      dump.Fetch("state", state);
      dump.Fetch("time", t0);
      dump.Fetch("lastVtkOutput", lastVtkOutput);
      dump.Fetch("lastDmpOutput", lastDmpOutput);
      dump.Fetch("lastTimevar", lastTimevar);
      dump.Fetch("nvtk", nvtk);
      dump.Fetch("ndmp", ndmp);
      astra::cout << "Main: Restart at time t=" << t0 << std::endl;
    } else {
      ///////////////////////////////
      // Initial condition generation
      //////////////////////////////

      initFlow.Init(state);
      // Write initial condition
      if(timevarStep>=0.0) {
        timevarOutput.Reset();
        timevarOutput.Write(state, 0.0);
      }

      if(outputVtkStep>=0) {
        WriteFile(input, &grid, state, nvtk, 0.0);
        nvtk++;
      }

      // Write initial dump if requested
      if(outputDmpStep>0.0) {
        Dump dump(&grid, outputDmpDir+"/"+CreateDumpFileName(ndmp));
        dump.Register("state", state);
        dump.Register("time", 0.0);
        dump.Register("lastVtkOutput", lastVtkOutput);
        dump.Register("lastDmpOutput", lastDmpOutput);
        dump.Register("lastTimevar", lastTimevar);
        dump.Register("nvtk", nvtk);
        dump.Register("ndmp", ndmp);
        dump.Write();
        ndmp++;
      }
    }


    astra::cout << "Main: Starting time integration..." << std::endl;
    // Init the time integrator
    auto timeIntegrator = TimeIntegratorFactory<Array3D<complex>>::Create(input, &grid, rhsVector);
    timeIntegrator->SetTime(t0);
    real tstop = input.Get<real>("TimeIntegrator","tstop",0);
    Kokkos::Timer timer;
    while(timeIntegrator->GetTime() < tstop) {
      timeIntegrator->SetDtMax(tstop - timeIntegrator->GetTime());
      timeIntegrator->Cycle(state);

      // Output
      if(timeIntegrator->GetTime()-lastVtkOutput>=outputVtkStep) {
        WriteFile(input, &grid, state, nvtk, timeIntegrator->GetTime());
        nvtk++;
        lastVtkOutput += outputVtkStep;
      }
      if(timevarStep>0.0 && timeIntegrator->GetTime()-lastTimevar>=timevarStep) {
        timevarOutput.Write(state, timeIntegrator->GetTime());
        lastTimevar += timevarStep;
      }
      if(outputDmpStep>0.0 && timeIntegrator->GetTime()-lastDmpOutput>=outputDmpStep) {
        Dump dump(&grid, outputDmpDir+"/"+CreateDumpFileName(ndmp));
        ndmp++;
        lastDmpOutput += outputDmpStep;
        dump.Register("state", state);
        dump.Register("time", timeIntegrator->GetTime());
        dump.Register("lastVtkOutput", lastVtkOutput);
        dump.Register("lastDmpOutput", lastDmpOutput);
        dump.Register("lastTimevar", lastTimevar);
        dump.Register("nvtk", nvtk);
        dump.Register("ndmp", ndmp);
        dump.Write();
    }

      // End?
      if(input.CheckForAbort() || timeIntegrator->CheckForMaxRuntime()) {
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


