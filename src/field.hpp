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
    // Constructor with a name and dimensions. Arrays will be subsequently added with Add()
    Field(std::string name, std::array<int,T::rank> n) : name(name), np(n) {};

    // Constructor with a name and a Field. The new field will have the same dimensions and variables as the input field.
    template<typename G>
    Field(std::string name, Field<G>& in) : name(name), np(in.GetDimensions()) {
      for(auto& it : in) {
        this->Add(it.first);
      }
    }
    // Constructor with a name, dimenions and a Field. The new field will have the new dimensions but the same variables as the input field.
    template<typename G>
    Field(std::string name, std::array<int,T::rank> n, Field<G>& in) : name(name), np(n) {
      for(auto& it : in) {
        this->Add(it.first);
      }
    }

    void Add(std::string key) {
      if(map.count(key)) {
        throw std::runtime_error(key+" already exists in the Field \""+name+"\"");
      }
      // allocate memory
      map[key] = T("Field "+name+" "+key, this->np);
    }

    void Reset() {
      for(auto& it : map) {
        auto view = it.second;
        Kokkos::deep_copy(view, 0.0);
      }
    }

    template<typename G>
    void CopyFrom(Field<G>& in) {
      for(auto& it : in) {
        Kokkos::deep_copy(map.at(it.first), in[it.first]);
      }
    }


  // Accessors to make it behave like a map (but throw exception if key not found)
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
    std::array<int,T::rank> GetDimensions() {return np;};
  private:
    std::string name;
    std::map<std::string,T> map;
    std::array<int,T::rank> np;          ///< local process number of grid points in real space
};

#endif