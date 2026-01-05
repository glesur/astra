// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic interface for the right hand side computation

#ifndef RIGHTHANDSIDE_HPP_
#define RIGHTHANDSIDE_HPP_

#include "field.hpp"
#include "input.hpp"

class Grid;

template <typename T>
class RightHandSideConcept {
  public:
    virtual ~RightHandSideConcept() = default;
    virtual void ExplicitStep(Field<T>& fldin, Field<T>& dfld, real t) = 0;
    virtual void ImplicitStep(Field<T>& fldin, real t, real dt) = 0;
    virtual void SetTinit(real t0) = 0;
    virtual real GetInvDt() = 0;
    virtual std::vector<std::string> GetVariables() = 0;
};

template <typename T, typename Shear>
class RightHandSide : public RightHandSideConcept<T> {
  public:
    RightHandSide(Input &input, Grid *grid) : grid(grid), shear(input, grid) {}
    virtual ~RightHandSide() {}

    virtual void ExplicitStep(Field<T>& fldin, Field<T>& dfld, real t) = 0;
    virtual void ImplicitStep(Field<T>& fldin, real t, real dt) = 0;
    void SetTinit(real t0) {
      this->shear.SetTinit(t0);
    }
    virtual real GetInvDt() = 0;
    virtual std::vector<std::string> GetVariables() = 0;

  protected:
    Grid *grid{nullptr};
    Shear shear;
};

#endif // RIGHTHANDSIDE_HPP_