// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// RK2 time integrator
#ifndef RK2_HPP_
#define RK2_HPP_

#include "timeIntegrator.hpp"

// Implementation of IDEFIX second order TVD time integrator
template <typename T>
class RK2TimeIntegrator : public TimeIntegrator<T> {
  public:
    RK2TimeIntegrator(Input &input, Grid *grid, std::vector<RightHandSide<T>*> rhsVector) : TimeIntegrator<T>(input,grid,rhsVector), dfld("RK2 dfld",grid->npf), fld0("RK2 fld0",grid->npf) {
      // Init dfld vector for the rhs
      for(auto rhs : rhsVector) {
        for(auto var : rhs->GetVariables()) {
          dfld.Add(var);
          fld0.Add(var);
        }
      }
    }

    ~RK2TimeIntegrator() {}

    void Cycle(Field<T>& fld) override {
      astra::pushRegion("RK2TimeIntegrator::Cycle");
      // Implement the RK2 time integration step here

      // Explicit part
      this->dt = 0.0;

      // Save the initial field
      fld0.CopyFrom(fld);
      ///////////////////////////////////////:
      // First Stage
      ///////////////////////////////////////

      dfld.Reset();
      for(auto rhs : this->rhsVector) {
        rhs->ExplicitStep(fld, dfld, this->t);
        this->dt += rhs->GetInvDt();
      }
      // Get the timestep
      this->dt = this->cfl/this->dt;;

      // Update the field
      for(auto& it : fld) {
        auto view = it.second;
        auto dview = dfld[it.first];
        real dt = this->dt;
        astra_for("rk2_update1_"+it.first,fld,
          KOKKOS_LAMBDA(int i,int j,int k) {
            view(i,j,k) += dt * dview(i,j,k);
        }); 
      }

      // Implicit part
      for(auto rhs : this->rhsVector) {
        rhs->ImplicitStep(fld, this->t, this->dt);
      }

      ///////////////////////////////////////:
      // Second Stage
      ///////////////////////////////////////
      dfld.Reset();
      real stageTime = this->t + this->dt;

      for(auto rhs : this->rhsVector) {
        rhs->ExplicitStep(fld, dfld, stageTime);
      }

      // Update the field
      for(auto& it : fld) {
        auto view = it.second;
        auto dview = dfld[it.first];
        real dt = this->dt;
        astra_for("rk2_update2_"+it.first,fld,
          KOKKOS_LAMBDA(int i,int j,int k) {
            view(i,j,k) += dt *  dview(i,j,k);
        }); 
      }

      // Implicit part
      for(auto rhs : this->rhsVector) {
        rhs->ImplicitStep(fld, stageTime, this->dt);
      }

      ///////////////////////////////////////:
      // Final combination
      ///////////////////////////////////////

      for(auto& it : fld) {
        auto view = it.second;
        auto view0 = fld0[it.first];
        real dt = this->dt;
        astra_for("rk2_update2_"+it.first,fld,
          KOKKOS_LAMBDA(int i,int j,int k) {
            view(i,j,k) = 0.5 * (view0(i,j,k) + view(i,j,k));
        });
      }

      // Call base class cycle to update time and cycle count
      astra::popRegion();
      TimeIntegrator<T>::Cycle(fld);
    }

  private:
    Field<T> dfld; // a temporary field to store the time derivative
    Field<T> fld0; // a temporary field to store the original field

};

#endif // RK2_HPP_