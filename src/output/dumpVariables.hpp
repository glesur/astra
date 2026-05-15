// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef OUTPUT_DUMP_VARIABLES_HPP_
#define OUTPUT_DUMP_VARIABLES_HPP_

#include <string>
#include <map>
#include "field.hpp"
class Dump;

class DumpVariables {
  public:
    DumpVariables() = default;

    static void Register(std::string name, Field<Array3D<complex>> &field);
    static void Register(std::string name, real &value);
    static void Register(std::string name, int &value);

    static void StoreToDump(Dump *dump);
    static void FetchFromDump(Dump *dump);

  private:
    static std::map<std::string, Field<Array3D<complex>>> fields;
    static std::map<std::string, real*> reals;
    static std::map<std::string, int*> ints;
};

#endif// OUTPUT_DUMP_VARIABLES_HPP_