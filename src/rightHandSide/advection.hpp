// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A purely advective right hand side along a given direction

#ifndef ADVECTION_HPP_
#define ADVECTION_HPP_

#include "rightHandSide.hpp"
#include "input.hpp"
#include "arrays.hpp"

class Grid;

class Advection : public RightHandSide<Array3D<complex>> {
  public:
    Advection(Input &input, Grid *grid);

    ~Advection();

    void ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) override;

    void ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) override;

    real GetInvDt() override;

    std::vector<std::string> GetVariables() override;

  private:
    int direction{0};
    real velocity{1.0};
};

#endif // ADVECTION_HPP_