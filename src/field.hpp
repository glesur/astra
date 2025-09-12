// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FIELD_HPP_
#define FIELD_HPP_

#include <vector>
#include <memory>
#include "arrays.hpp"

template<typename T>
class Field {
  public:
    Field(std::string name, std::array<int,T::rank> n) : name(name), npr(n) {};

    void Add(std::string key) {
      if(map.count(key)) {
        throw std::runtime_error(key+" already exists in the Field \""+name+"\"");
      }
      // allocate memory
      map[key] = T("Field "+name+" "+key, this->npr);
    }

    T & operator[] (std::string var) {
      return this->map.at(var);
    }

    T operator[] (std::string var) const {
      return this->map.at(var);
    }

    // iterators through the fields
    using iterator = typename std::map<std::string, T>::iterator;
    using const_iterator = typename std::map<std::string, T>::const_iterator;

    iterator begin() { return map.begin(); }
    iterator end() { return map.end(); }

    const_iterator begin() const { return map.begin(); }
    const_iterator end() const { return map.end(); }

    static constexpr int rank{T::rank};
    std::array<<int,T::rank> GetDimensions() {return npr};
  private:
    std::string name;
    std::map<std::string,T> map;
    std::array<int,T::rank> npr;          ///< local process number of grid points in real space
};

#endif