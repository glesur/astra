// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************


#ifndef TEST_HPP_
#define TEST_HPP_

class Input;

class Test {
 public:
  explicit Test(Input *input);
  void Run();
  void Finalize();
 private:
  Input* input;
};

#endif // TEST_HPP_
