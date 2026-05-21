// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// A generic Factory for the right hand side computation

#ifndef RIGHTHANDSIDE_RIGHTHANDSIDEFACTORY_HPP_
#define RIGHTHANDSIDE_RIGHTHANDSIDEFACTORY_HPP_

#include <string>
#include <utility>
#include <vector>
#include "rightHandSide.hpp"
#include "grid.hpp"
#include "advection.hpp"
#include "hydro.hpp"
#include "mhd.hpp"
#include "input.hpp"
#include "shear.hpp"

// A factory for right hand side computations
template<typename T>
class RightHandSideFactory {
 public:
  enum class ShearType { NoShear, LinearShear };
  static std::vector<std::unique_ptr<RightHandSideConcept<T>>> Create(Input &input, Grid *grid) {
    std::vector<std::unique_ptr<RightHandSideConcept<T>>> rhsVector;

    int numRhs = input.CheckEntry("Physics","rhs");
    if(numRhs < 1) {
      throw std::runtime_error("No right hand side specified in the input file [Physics]:rhs");
    }
    std::string shearTypeStr = input.GetOrSet<std::string>("Physics","shear_type",0,"disabled");
    ShearType shearType;
    if(shearTypeStr =="disabled") {
      shearType = ShearType::NoShear;
    } else if(shearTypeStr =="linear") {
      shearType = ShearType::LinearShear;
    } else {
      throw std::runtime_error("Unknown shear type: " + shearTypeStr);
    }

    for(int irhs = 0 ; irhs < numRhs ; irhs++) {
      std::string method = input.Get<std::string>("Physics","rhs",irhs);
      if(method == "advection") {
        if(shearType == ShearType::NoShear) {
          rhsVector.emplace_back(std::move(std::make_unique<Advection<NoShear>>(input, grid)));
        } else if(shearType == ShearType::LinearShear) {
          rhsVector.emplace_back(std::move(std::make_unique<Advection<LinearShear>>(input, grid)));
        }
      } else if(method == "hydro") {
        if(shearType == ShearType::NoShear) {
          rhsVector.emplace_back(std::move(std::make_unique<Hydro<NoShear>>(input, grid)));
        } else if(shearType == ShearType::LinearShear) {
          rhsVector.emplace_back(std::move(std::make_unique<Hydro<LinearShear>>(input, grid)));
        }
      } else if(method == "mhd") {
        if(shearType == ShearType::NoShear) {
          rhsVector.emplace_back(std::move(std::make_unique<Mhd<NoShear>>(input, grid)));
        } else if(shearType == ShearType::LinearShear) {
          rhsVector.emplace_back(std::move(std::make_unique<Mhd<LinearShear>>(input, grid)));
        }
      } else {
        throw std::runtime_error("Unknown right hand side method: " + method);
      }
    }
    return rhsVector;
  }
};

#endif // RIGHTHANDSIDE_RIGHTHANDSIDEFACTORY_HPP_
