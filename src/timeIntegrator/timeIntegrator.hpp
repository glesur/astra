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
#include "rightHandSide.hpp"
#include "logger.hpp"

template <typename T>
class TimeIntegrator {
  public:
    TimeIntegrator(Input &input, Grid *grid, std::vector<RightHandSide<T>*> rhsVector) : rhsVector(rhsVector), grid(grid), logger(input, grid, this) {
      logger.Start();
    }
    virtual ~TimeIntegrator() {}

    virtual void Cycle(Field<T>& field) {
      t=t+dt;
      ncycles++;
      logger.Show(ncycles);
    };
    real GetTime() { return t; }
    real GetTimeStep() { return dt; }
    int GetCycle() { return ncycles; }
    
  protected:
    real t{0.0};
    real dt{0.0};
    int ncycles{0};
    Grid *grid;
    std::vector<RightHandSide<T>*> rhsVector;
    Logger<T> logger;
};

#endif // TIMEINTEGRATOR_HPP_