// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic interface to initialise the flow
#ifndef INITFLOW_INITFLOW_HPP_
#define INITFLOW_INITFLOW_HPP_

#include "input.hpp"
#include "field.hpp"
#ifdef WITH_PYTHON
  #include "astrapy.hpp"
#endif

class Grid;


class InitFlow {
 public:
  InitFlow(Input &input, Grid *grid);

  void Init(Field<Array3D<complex>>& field);

 private:
  Grid *grid;
  Input *input;
  void ShearLayer(Field<ArrayHost3D<complex>>& field);
  void MeanFlow(Field<ArrayHost3D<complex>>& field);
  void MeanField(Field<ArrayHost3D<complex>>& field);
  void PureSine(Field<ArrayHost3D<complex>>& field);
  void LargeScale3DNoise(Field<ArrayHost3D<complex>>& field);
  void LargeScale2DNoise(Field<ArrayHost3D<complex>>& field);
  void LargeScale1DNoise(Field<ArrayHost3D<complex>>& field);
  void Projector(Field<ArrayHost3D<complex>>& field);

  std::array<ArrayHost1D<real>,3> kx;
  std::array<ArrayHost1D<real>,3> x;

  #ifdef WITH_PYTHON
    AstraPy astraPy;
  #endif
};

#endif // INITFLOW_INITFLOW_HPP_
