// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// Euler time integrator
#ifndef TIMEINTEGRATOR_EULER_HPP_
#define TIMEINTEGRATOR_EULER_HPP_

#include <algorithm>
#include <vector>
#include "timeIntegrator.hpp"

template <typename T>
class EulerTimeIntegrator : public TimeIntegrator<T> {
 public:
  EulerTimeIntegrator(Input &input, Grid *grid, std::vector<std::unique_ptr<RightHandSideConcept<T>>> &rhsVector)
    : TimeIntegrator<T>(input,grid,rhsVector), dfld("Euler dfld",grid->npf) {

    // Init dfld vector for the rhs
    for(auto &rhs : rhsVector) {
      for(auto var : rhs->GetVariables()) {
        dfld.Add(var);
      }
    }
  }

  ~EulerTimeIntegrator() {}

  void Cycle(Field<T>& field) override {
    astra::pushRegion("EulerTimeIntegrator::Cycle");
    // Implement the Euler time integration step here
    dfld.Reset();
    // Explicit part

    this->dt = 0.0;
    for(auto rhs : this->rhsVector) {
      rhs->ExplicitStep(field, dfld, this->t);
      this->dt += rhs->GetInvDt();
    }
    // Get the timestep
    this->dt = std::min(this->cfl/this->dt, this->dtMax);

    // Update the field
    for(auto& it : field) {
      auto view = it.second;
      auto dview = dfld[it.first];
      real dt = this->dt;
      astra_for("euler_update_"+it.first,field,
        KOKKOS_LAMBDA(int64_t i,int64_t j,int64_t k) {
          view(i,j,k) += dt * dview(i,j,k);
      });
    }

    // Implicit part
    for(auto rhs : this->rhsVector) {
      rhs->ImplicitStep(field, this->t, this->dt);
    }
    // Call base class cycle to update time and cycle count
    TimeIntegrator<T>::Cycle(field);

    // Post stage operations
    for(auto rhs : this->rhsVector) {
      rhs->PostStage(field, this->t);
    }
    astra::popRegion();
  }

 private:
  Field<T> dfld; // a temporary field to store the time derivative
};

#endif // TIMEINTEGRATOR_EULER_HPP_
