// ***********************************************************************************
// ASTRA spectral code
// Accelerated Spectral code for TuRbulent plasmA
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

//#ifdef WITH_MPI

#ifndef MPI_HPP_
#define MPI_HPP_
#include <mpi.h>

//redefined MPI complex type to match Astra Complex type
#define MPI_Astra_Complex  MPI_C_DOUBLE_COMPLEX
#define MPI_Astra_real  MPI_DOUBLE


#endif // MPI_HPP_
//#endif // WITH_MPI