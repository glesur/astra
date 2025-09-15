// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// Euler time integrator
#ifndef EULER_HPP_
#define EULER_HPP_

#include "timeIntegrator.hpp"

template <typename T>
class EulerTimeIntegrator : public TimeIntegrator<T> {
  public:
    EulerTimeIntegrator(Input &input, Grid *grid, std::vector<RightHandSide<T>*> rhsVector) : TimeIntegrator<T>(input,grid,rhsVector), dfld("Euler dfld",grid->npf) {
      // Init dfld vector for the rhs
      for(auto rhs : rhsVector) {
        for(auto var : rhs->GetVariables()) {
          dfld.Add(var);
        }
      }
    }

    ~EulerTimeIntegrator() {}

    void Cycle(Field<T>& field) override {
      // Implement the Euler time integration step here
      dfld.Reset();
      // Explicit part
      real cmax = 0;
      this->dt = 0.0;
      for(auto rhs : this->rhsVector) {
        rhs->ExplicitStep(field, dfld);
        this->dt += rhs->GetInvDt();
      }
      // Get the timestep
      this->dt = 1/this->dt;;

      // Update the field
      for(auto& it : field) {
        auto view = it.second;
        auto dview = dfld[it.first];
        real dt = this->dt;
        astra_for("euler_update_"+it.first,field,
          KOKKOS_LAMBDA(int i,int j,int k) {
            view(i,j,k) += dt * dview(i,j,k);
        }); 
      }

      // Implicit part
      for(auto rhs : this->rhsVector) {
        rhs->ImplicitStep(field, this->dt);
      }
      // Call base class cycle to update time and cycle count
      TimeIntegrator<T>::Cycle(field);
    }

  private:
    Field<T> dfld; // a temporary field to store the time derivative
  
};

#endif // EULER_HPP_