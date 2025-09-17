// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef HYDRO_HPP_
#define HYDRO_HPP_

#include "rightHandSide.hpp"
#include "input.hpp"
#include "arrays.hpp"

class Grid;

// A class for the hydrodynamics right hand side
class Hydro : public RightHandSide<Array3D<complex>> {
public:
  Hydro(Input &input, Grid *grid);
  ~Hydro();

  void ExplicitStep(Field<Array3D<complex>>& fldin, Field<Array3D<complex>>& dfld, real t) override;
  void ImplicitStep(Field<Array3D<complex>>& fldin, real t, real dt) override;

  void Projector(Field<Array3D<complex>>& fldin);

  real GetInvDt() override;
  std::vector<std::string> GetVariables() override;

private:
  real nu;
  Array3D<real> vr1, vr2, vr3;
  Array3D<real> wr11,wr12,wr13,wr22,wr23,wr33;
  Array3D<complex> wf11,wf12,wf13,wf22,wf23,wf33;
  std::array<int,3> npr, npf;


};

#endif // HYDRO_HPP_