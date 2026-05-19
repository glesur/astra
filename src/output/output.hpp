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
#include "linearshear.hpp"

class TimeVarOutput;
class Slice;
class Output {
  friend class Slice;
  public:
    Output(Input &input, Grid &grid);
    ~Output(); // needed by the incomplete type TimeVarOutput used in a unique_ptr. 
    void CheckForOutput(Input &input, Field<Array3D<complex>> state, real time);
    void ForceDump(real time);
    void RestartFromDump(Input &input);

    // This function is technically private, but needs to be public for Cuda lambda captures.
    void ComputeCurl(std::array<Array3D<complex>, 3> out , std::array<Array3D<complex>, 3> in, real time);
    
  private:
    int GetLastDumpInDirectory(std::string directory_str);  
    std::string CreateDumpFileName(int n);
    void ComputeAdditionalVariables(Field<Array3D<complex>> state, real time);  // Compute any additional variables that we want to output but are not part of the state vector

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
    bool haveShear{false};
    std::unique_ptr<LinearShear> linearShear;

    // Vtk output
    real lastVtkOutput{0.0};
    real outputVtkStep{0.1};
    std::string outputVtkDirectory{"./"};
    std::vector<std::string> vtkAdditionalVariableNames;
    Field<Array3D<complex>> vtkAdditionalVariables;  // Additional variables to output in vtk files
    int nvtk{0};

    // Slices outputs
    bool haveSlices{false};
    std::vector<std::unique_ptr<Slice>> slices;

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