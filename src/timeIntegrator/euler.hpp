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
    EulerTimeIntegrator(std::vector<RightHandSide<T>*> rhsVector, Field<T>& field) : TimeIntegrator<T>(rhsVector), dfld(field.Clone("Euler dfld")) {}
    ~EulerTimeIntegrator() {}

    void Cycle(Field<T>& field) override {
      // Implement the Euler time integration step here
      dfld.Reset();
      // Explicit part
      for(auto rhs : this->rhsVector) {
        rhs->ExplicitStep(field, dfld);
      }

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
    }

  private:
    Field<T> dfld; // a temporary field to store the time derivative
  
};

#endif // EULER_HPP_