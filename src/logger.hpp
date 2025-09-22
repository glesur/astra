// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <string>
#include <iomanip>

#include "grid.hpp"
#include "input.hpp"
template<typename T> class TimeIntegrator;


template <typename T>
class Logger {
  public:
    Logger(Input &input, Grid *grid, TimeIntegrator<T> *timeIntegrator) : grid(grid), timeIntegrator(timeIntegrator) {
      timer.reset();
      lastLog = timer.seconds();
      cyclePeriod = input.GetOrSet<int>("Output","log",0,10);
      //isSilent = input.GetOrSet<bool>("Logger","silent",0,false);
    }
    void Show(int);
    void Start();

  private:
    TimeIntegrator<T> *timeIntegrator{nullptr};
    const int col_width{16};
    Kokkos::Timer timer;
    double lastLog{0.0};
    int cyclePeriod{10};
    bool isSilent{false};
    Grid *grid{nullptr};
    Input *input{nullptr};
};

#include "timeIntegrator.hpp"

template <typename T>
void Logger<T>::Start() {
  timer.reset();
  lastLog = timer.seconds();
  astra::cout << "Logger: ";
  astra::cout << std::setw(col_width) << "time";
  astra::cout << " | " << std::setw(col_width) << "cycle";
  astra::cout << " | " << std::setw(col_width) << "time step";
  astra::cout << " | " << std::setw(col_width) << "cell (updates/s)";
#ifdef WITH_MPI
  //astra::cout << " | " << std::setw(col_width) << "MPI overhead (%)";
#endif
  astra::cout << std::endl;
}

template <typename T>
void Logger<T>::Show(int ncycles) {
  if(isSilent) return;
  if(ncycles % cyclePeriod != 0 && ncycles != 0) return;

  double rawperf = (timer.seconds()-lastLog)/grid->npr[IDIR]/grid->npr[IDIR]
                      /grid->npr[IDIR]/cyclePeriod;


  lastLog = timer.seconds();

  astra::cout << "Logger: ";
  astra::cout << std::scientific;
  astra::cout << std::setw(col_width) << timeIntegrator->GetTime();
  astra::cout << " | " << std::setw(col_width) << ncycles;
  astra::cout << " | " << std::setw(col_width) << timeIntegrator->GetTimeStep();
  if(ncycles>=cyclePeriod) {
    astra::cout << " | " << std::setw(col_width) << 1 / rawperf;
#ifdef WITH_MPI
  //astra::cout << std::fixed;
  //astra::cout << " | " << std::setw(col_width) << mpiOverhead;
#endif
  } else {
    astra::cout << " | " << std::setw(col_width) << "N/A";
#ifdef WITH_MPI
   // astra::cout << " | " << std::setw(col_width) << "N/A";
#endif
  }
  astra::cout << std::endl;
}

#endif // LOGGER_HPP_