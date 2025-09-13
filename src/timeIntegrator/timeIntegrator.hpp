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

template <typename T>
class TimeIntegrator {
  public:
    TimeIntegrator(std::vector<RightHandSide<T>*> rhsVector) : rhsVector(rhsVector) {}
    virtual ~TimeIntegrator() {}

    virtual void Cycle(Field<T>& field) = 0;

  protected:
    int ncycle{0};
    double time{0.0};
    double dt{0.0};
    std::vector<RightHandSide<T>*> rhsVector;
};
#endif // TIMEINTEGRATOR_HPP_