// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic interface to initialise the flow
#ifndef INITFLOW_HPP_
#define INITFLOW_HPP_

#include "input.hpp"
#include "field.hpp"

class Grid;


class InitFlow {
public:
  InitFlow(Input &input, Grid *grid);
  
  void Init(Field<Array3D<complex>>& field);

private:
  Grid *grid;
  Input *input;
  void MeanField(Field<ArrayHost3D<complex>>& field);
  void LargeScaleNoise(Field<ArrayHost3D<complex>>& field);
  void Projector(Field<ArrayHost3D<complex>>& field);

  std::array<ArrayHost1D<real>,3> kx;
  std::array<ArrayHost1D<real>,3> x;

};

#endif // INITFLOW_HPP_