// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic Factory for the right hand side computation

#ifndef RIGHTHANDSIDEFACTORY_HPP_
#define RIGHTHANDSIDEFACTORY_HPP_

#include <string>
#include <vector>
#include "rightHandSide.hpp"
#include "advection.hpp"
#include "hydro.hpp"
#include "input.hpp"
#include "grid.hpp"

// A factory for right hand side computations
template<typename T>
class RightHandSideFactory {
public:
  static std::vector<RightHandSide<T>*> Create(Input &input, Grid *grid) {
    std::vector<RightHandSide<T>*> rhsVector;

    int numRhs = input.CheckEntry("Physics","rhs");
    if(numRhs < 1) {
      throw std::runtime_error("No right hand side specified in the input file [Physics]:rhs");
    }

    for(int irhs = 0 ; irhs < numRhs ; irhs++) {
      std::string method = input.Get<std::string>("Physics","rhs",irhs);
      if(method == "advection") {
        rhsVector.push_back(new Advection(input, grid));
      } else if(method == "hydro") {
        rhsVector.push_back(new Hydro(input, grid));
      } else {
        throw std::runtime_error("Unknown right hand side method: " + method);
      }
    }
    return rhsVector;
  }
};

#endif // RIGHTHANDSIDEFACTORY_HPP_
