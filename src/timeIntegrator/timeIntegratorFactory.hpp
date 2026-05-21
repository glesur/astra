// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef TIMEINTEGRATOR_TIMEINTEGRATORFACTORY_HPP_
#define TIMEINTEGRATOR_TIMEINTEGRATORFACTORY_HPP_

#include <memory>
#include <string>
#include <vector>
#include "timeIntegrator.hpp"
#include "rk3.hpp"
#include "rk2.hpp"
#include "euler.hpp"
#include "input.hpp"
#include "rightHandSide.hpp"

// A factory for time integrators
template<typename T>
class TimeIntegratorFactory {
 public:
  static std::unique_ptr<TimeIntegrator<T>> Create(Input &input, Grid *grid, std::vector<std::unique_ptr<RightHandSideConcept<T>>> &rhsVector) {
    std::string method = input.Get<std::string>("TimeIntegrator","method",0);
    if(method == "rk3") {
      return std::make_unique<RK3TimeIntegrator<T>>(input, grid, rhsVector);
    } else if(method == "rk2") {
      return std::make_unique<RK2TimeIntegrator<T>>(input, grid, rhsVector);
    } else if(method == "euler") {
      return std::make_unique<EulerTimeIntegrator<T>>(input, grid, rhsVector);
    } else {
      throw std::runtime_error("Unknown time integrator method: " + method);
    }
  }
};

#endif // TIMEINTEGRATOR_TIMEINTEGRATORFACTORY_HPP_
