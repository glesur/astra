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
class RightHandSide {
  public:
    RightHandSide(Input &input, Grid *grid) : grid(grid) {}
    virtual ~RightHandSide() {}

    virtual void ExplicitStep(Field<T>& fldin, Field<T>& dfld, real t) = 0;
    virtual void ImplicitStep(Field<T>& fldin, real t, real dt) = 0;
    virtual real GetInvDt() = 0;
    virtual std::vector<std::string> GetVariables() = 0;

  protected:
    Grid *grid{nullptr};
};

#endif // RIGHTHANDSIDE_HPP_