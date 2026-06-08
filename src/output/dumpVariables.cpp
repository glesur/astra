// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <map>
#include <string>
#include "field.hpp"
#include "dumpVariables.hpp"
#include "dump.hpp"

std::map<std::string, int*> DumpVariables::ints;
std::map<std::string, real*> DumpVariables::reals;
std::map<std::string, Field<Array3D<complex>>> DumpVariables::fields;

void DumpVariables::Register(std::string name, Field<Array3D<complex>> &field) {
  if(fields.find(name) != fields.end()) {
    throw std::runtime_error("Error: variable \"" + name + "\" is already registered in DumpVariables.");
  }
  fields[name] = field;
}

void DumpVariables::Register(std::string name, real &value) {
  if(reals.find(name) != reals.end()) {
    throw std::runtime_error("Error: variable \"" + name + "\" is already registered in DumpVariables.");
  }
  reals[name] = &value;
}

void DumpVariables::Register(std::string name, int &value) {
  if(ints.find(name) != ints.end()) {
    throw std::runtime_error("Error: variable \"" + name + "\" is already registered in DumpVariables.");
  }
  ints[name] = &value;
}

void DumpVariables::StoreToDump(Dump *dump) {
  for(const auto& pair : fields) {
    dump->Register(pair.first, pair.second);
  }
  for(const auto& pair : reals) {
    dump->Register(pair.first, *(pair.second));
  }
  for(const auto& pair : ints) {
    dump->Register(pair.first, *(pair.second));
  }
}

void DumpVariables::FetchFromDump(Dump *dump) {
  for(auto& pair : fields) {
    dump->Fetch(pair.first, pair.second);
  }
  for(const auto& pair : reals) {
    dump->Fetch(pair.first, *(pair.second));
  }
  for(const auto& pair : ints) {
    dump->Fetch(pair.first, *(pair.second));
  }
}

void DumpVariables::Release() {
  fields.clear();
  reals.clear();
  ints.clear();
}
