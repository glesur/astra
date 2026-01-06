// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// RK3 time integrator
#ifndef RK3_HPP_
#define RK3_HPP_

#include "timeIntegrator.hpp"

template <typename T>
class RK3TimeIntegrator : public TimeIntegrator<T> {
  public:
    RK3TimeIntegrator(Input &input, Grid *grid, std::vector<std::unique_ptr<RightHandSideConcept<T>>> &rhsVector) : TimeIntegrator<T>(input,grid,rhsVector), dfld("RK3 dfld",grid->npf), dfld1("RK3 dfld",grid->npf) {
      // Init dfld vector for the rhs
      for(auto &rhs : rhsVector) {
        for(auto var : rhs->GetVariables()) {
          dfld.Add(var);
          dfld1.Add(var);
        }
      }

    }

    ~RK3TimeIntegrator() {}

    void Cycle(Field<T>& fld) override {
      astra::pushRegion("RK3TimeIntegrator::Cycle");
      // Implement the RK3 time integration step here

      // Explicit part

      this->dt = 0.0;

      ///////////////////////////////////////:
      // First Stage
      ///////////////////////////////////////

      dfld.Reset();
      for(auto rhs : this->rhsVector) {
        rhs->ExplicitStep(fld, dfld, this->t);
        this->dt += rhs->GetInvDt();
      }
      // Get the timestep
      this->dt = std::min(this->cfl/this->dt, this->dtMax);

      // Update the field
      for(auto& it : fld) {
        auto view = it.second;
        auto dview = dfld[it.first];
        real dt = this->dt;
        const real gamma = gammaRK[0];
        astra_for("rk3_update1_"+it.first,fld,
          KOKKOS_LAMBDA(int i,int j,int k) {
            view(i,j,k) += gamma * dt * dview(i,j,k);
        }); 
      }

      // Implicit part
      for(auto rhs : this->rhsVector) {
        rhs->ImplicitStep(fld, this->t, this->dt*this->gammaRK[0]);
      }

      ///////////////////////////////////////:
      // Second Stage
      ///////////////////////////////////////
      dfld1.Reset();
      real stageTime = this->t + this->dt * this->gammaRK[0];

      for(auto rhs : this->rhsVector) {
        rhs->ExplicitStep(fld, dfld1, stageTime);
      }

      // Update the field
      for(auto& it : fld) {
        auto view = it.second;
        auto dview1 = dfld1[it.first];
        auto dview = dfld[it.first];
        real dt = this->dt;
        const real gamma = gammaRK[1];
        const real xi = xiRK[0];
        astra_for("rk3_update2_"+it.first,fld,
          KOKKOS_LAMBDA(int i,int j,int k) {
            view(i,j,k) += dt * (gamma * dview1(i,j,k) + xi * dview(i,j,k));
        }); 
      }

      // Implicit part
      for(auto rhs : this->rhsVector) {
        rhs->ImplicitStep(fld, stageTime, this->dt*(this->gammaRK[1]+this->xiRK[0]));
      }

      ///////////////////////////////////////:
      // Third Stage
      ///////////////////////////////////////
      dfld.Reset();
      stageTime = this->t + this->dt * (this->gammaRK[0]+this->gammaRK[1]+this->xiRK[0]);

      for(auto rhs : this->rhsVector) {
        rhs->ExplicitStep(fld, dfld, stageTime);
      }

      // Update the field
      for(auto& it : fld) {
        auto view = it.second;
        auto dview1 = dfld1[it.first];
        auto dview = dfld[it.first];
        real dt = this->dt;
        const real gamma = gammaRK[2];
        const real xi = xiRK[1];
        astra_for("rk3_update2_"+it.first,fld,
          KOKKOS_LAMBDA(int i,int j,int k) {
            view(i,j,k) += dt * (gamma * dview(i,j,k) + xi * dview1(i,j,k));
        }); 
      }

      // Implicit part
      for(auto rhs : this->rhsVector) {
        rhs->ImplicitStep(fld, stageTime, this->dt*(this->gammaRK[2]+this->xiRK[1]));
      }
      // Call base class cycle to update time and cycle count
      TimeIntegrator<T>::Cycle(fld);

      // Post stage operations
      for(auto rhs : this->rhsVector) {
        rhs->PostStage(fld, this->t);
      }

      astra::popRegion();
    }

  private:
    Field<T> dfld; // a temporary field to store the time derivative
    Field<T> dfld1; // a temporary field to store the time derivative
    const std::array<const real,3> gammaRK = {8.0 / 15.0 , 5.0 / 12.0 , 3.0 / 4.0};
    const std::array<const real,3> xiRK = {-17.0 / 60.0 , -5.0 / 12.0};
};

#endif // RK3_HPP_