// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************


// A generic interface for time integrators

#ifndef TIMEINTEGRATOR_HPP_
#define TIMEINTEGRATOR_HPP_

#include "field.hpp"
#include "grid.hpp"
#include "logger.hpp"
#include "rightHandSide.hpp"
#include "dumpVariables.hpp"
#include <limits>

#ifdef WITH_MPI
#include <mpi.h>
#endif

template <typename T>
class TimeIntegrator {
  public:
    TimeIntegrator(Input &input, Grid *grid, std::vector<std::unique_ptr<RightHandSideConcept<T>>> &rhsVector)  : grid(grid), logger(input, grid, this) {
      timer.reset();
      cfl = input.Get<real>("TimeIntegrator","cfl",0);
      maxRuntime = 3600*input.GetOrSet<double>("TimeIntegrator","max_runtime",0.0,-1.0);
      for (auto& rhs : rhsVector) {
        this->rhsVector.push_back(rhs.get());
      }
      DumpVariables::Register("time", t);
    }
    virtual ~TimeIntegrator() {}

    virtual void Cycle(Field<T>& field) {
      astra::pushRegion("TimeIntegrator::Cycle");
      t=t+dt;
      ncycles++;
      logger.Show(ncycles);
      astra::popRegion();
    };
    void Reset() { // Reset internal time proxies for the rhs (needed for remaps and restarts)
      for(auto &rhs : rhsVector) {
        rhs->SetTinit(t);
      }
    }
    real GetTime() { return t; }
    real GetTimeStep() { return dt; }
    void SetTime(real time) { 
      t = time; 
      Reset();
    }
    
    int GetCycle() { return ncycles; }
    void SetDtMax(real dtmax) { dtMax = dtmax; }

  bool CheckForMaxRuntime();  // Check whether maximum runtime has been reached

  protected:
    real t{0.0};
    real dt{0.0};
    real dtMax{std::numeric_limits<real>::max()};
    real cfl{1.0};
    int ncycles{0};
    Grid *grid;
    std::vector<RightHandSideConcept<T>*> rhsVector;
    Logger<T> logger;
    double maxRuntime;      // Maximum runtime requested (disabled when negative)
    Kokkos::Timer timer;    // Internal timer of the integrator
};

template <typename T>
bool TimeIntegrator<T>::CheckForMaxRuntime() {
    astra::pushRegion("TimeIntegrator::CheckForMaxRuntime");
    // if maxRuntime is negative, this function is disabled (default)
    if(this->maxRuntime < 0) {
      astra::popRegion();
      return(false);
    }

    double runtime = timer.seconds();
    bool runtimeReached{false};
  #ifdef WITH_MPI
    int runtimeValue = 0;
    if(runtime >= this->maxRuntime) runtimeValue = 1;
    MPI_Bcast(&runtimeValue, 1, MPI_INT, 0, MPI_COMM_WORLD);
    runtimeReached = runtimeValue > 0;
  #else
    runtimeReached = runtime >= this->maxRuntime;
  #endif
    if(runtimeReached) {
      astra::cout << "TimeIntegrator:CheckForMaxRuntime: Maximum runtime reached."
                << std::endl;
    }
    astra::popRegion();
    return(runtimeReached);
  }
#endif // TIMEINTEGRATOR_HPP_