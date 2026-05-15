// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef OUTPUT_OUTPUT_HPP_
#define OUTPUT_OUTPUT_HPP_


#include "grid.hpp"
#include "input.hpp"
#include "field.hpp"

class TimeVarOutput;

class Output {
  public:
    Output(Input &input, Grid &grid);
    ~Output(); // needed by the incomplete type TimeVarOutput used in a unique_ptr. 
    void CheckForOutput(Input &input, Field<Array3D<complex>> state, real time);
    void ForceDump(real time);
  private:
    int GetLastDumpInDirectory(std::string directory_str);  
    std::string CreateDumpFileName(int n);
    std::unique_ptr<TimeVarOutput> timeVarOutput;
    // Helper function to convert filesystem::file_time into std::time_t
    // see https://stackoverflow.com/questions/56788745/
    // This conversion "hack" is required in C++17 as no proper conversion bewteen
    // fs::last_write_time and std::time_t
    // exists in the standard library until C++20
    template <typename TP>
    std::time_t to_time_t(TP tp) {
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>
                    (tp - TP::clock::now() + std::chrono::system_clock::now());
        return std::chrono::system_clock::to_time_t(sctp);
    }

    Grid *grid;

    // Vtk output
    real lastVtkOutput{0.0};
    real outputVtkStep{0.1};
    std::string outputVtkDirectory{"./"};
    int nvtk{0};

    // Dump output
    real lastDmpOutput{0.0};
    real outputDmpStep{-1.0};
    std::string outputDmpDir{"./"};
    int ndmp{0};

    // Timevar output
    real lastTimevar{0.0};
    real timevarStep{-1.0};



};
#endif// OUTPUT_OUTPUT_HPP_