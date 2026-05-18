// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <map>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <limits.h>
#include <string>
#include <cstdio>
#include <sstream>
#include <iomanip>

#include "vtk.hpp"
#include "version.hpp"
#include "grid.hpp"
#include "global.hpp"
#include "arrays.hpp"
#include "bigEndian.hpp"

#include "shear.hpp"
#include "input.hpp"


/*init the object */
Vtk::Vtk(Grid *grid, Input &input, real time, std::string filebase, std::string directory) {

  // Initialise the root tag (used for MPI non-collective I/Os)
  this->isRoot = astra::prank == 0;
  this->grid = grid;

  /* Note that there are two kinds of dimensions:
     - nx1, nx2, nx3, derived from the grid, which are the global dimensions
     - nx1loc,nx2loc,n3loc, which are the local dimensions of the current datablock
  */

  // Create the coordinate array required in VTK files
  this->nx1 = grid->npr_glob[IDIR];
  this->nx2 = grid->npr_glob[JDIR];
  this->nx3 = grid->npr_glob[KDIR];

  this->nx1loc = grid->npr[IDIR];
  this->nx2loc = grid->npr[JDIR];
  this->nx3loc = grid->npr[KDIR];

  // Temporary storage on host for 3D arrays
  this->vect3D = new float[nx1loc*nx2loc*nx3loc];

  // Create MPI view when using MPI I/O
  #ifdef WITH_MPI
    int start[3];
    int size[3];
    int subsize[3];

    for(int dir = 0; dir < 3 ; dir++) {
      // VTK assumes Fortran array ordering, hence arrays dimensions are filled backwards
      start[2-dir] = 0;
      size[2-dir] = grid->npr_glob[dir];
      subsize[2-dir] = grid->npr[dir];
    }
    // Fix starting point along x
    start[2] = grid->npr[0]*astra::prank;
    
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, MPI_FLOAT, &this->view);
    MPI_Type_commit(&this->view);
  #endif // WITH_MPI

  // Check if we have shear

  fs::path outputDirectory = directory;
  std::string shearTypeStr = input.GetOrSet<std::string>("Physics","shear_type",0,"disabled");
  if(shearTypeStr =="linear") {
    this->haveShear = true;
    this->linearShear = std::make_unique<LinearShear>(input, grid);
  }

  if(isRoot) {
    if(!fs::is_directory(outputDirectory)) {
      try {
        if(!fs::create_directory(outputDirectory)) {
          std::stringstream msg;
          msg << "Cannot create directory " << outputDirectory << std::endl;
          throw std::runtime_error(msg.str());
        }
      } catch(std::exception &e) {
        std::stringstream msg;
        msg << "Cannot create directory " << outputDirectory << std::endl;
        msg << e.what();
        throw std::runtime_error(msg.str());
      }
    }
  }

  // Open file;
  fs::path filename;
  std::stringstream ssfileName;
  ssfileName << filebase << ".vtk";
  filename = outputDirectory/ssfileName.str();

  astra::cout << "Vtk: Writing file " << ssfileName.str() << "..." << std::flush;

  timer.reset();

  // Check if file exists, if yes, delete it
  if(this->isRoot) {
    if(fs::exists(filename)) {
      fs::remove(filename);
    }
  }


  // Open file and write header
  #ifdef WITH_MPI
    this->comm = MPI_COMM_WORLD;
    MPI_Barrier(this->comm);
    // Open file for creating, return error if file already exists.
    MPI_File_open(this->comm, filename.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_RDWR
                  | MPI_MODE_EXCL | MPI_MODE_UNIQUE_OPEN,
                  MPI_INFO_NULL, &this->fileHdl);
    this->offset = 0;
  #else
    fileHdl = fopen(filename.c_str(),"wb");

    if(this->fileHdl == NULL) {
      std::stringstream msg;
      msg << "Unable to open file " << filename << std::endl;
      msg << "Check that you have write access and that you don't exceed your quota." << std::endl;
      throw std::runtime_error(msg.str());
    }
  #endif

  

  // Write the header
  std::string header;
  std::stringstream ssheader;

  /* -------------------------------------------
  1. File version and identifier
  ------------------------------------------- */

  ssheader << "# vtk DataFile Version 2.0" << std::endl;

  /* -------------------------------------------
  2. Header
  ------------------------------------------- */

  ssheader << "Astra " << ASTRA_VERSION << " VTK Data" << std::endl;

  /* ------------------------------------------
  3. File format
  ------------------------------------------ */

  ssheader << "BINARY" << std::endl;

  /* ------------------------------------------
  4. Dataset structure
  ------------------------------------------ */

  ssheader << "DATASET STRUCTURED_POINTS" << std::endl;

  // fields: geometry, periodicity, time, 6 NativeCoordinates (x1l, x2l, x3l, x1c, x2c, x3c)
  int nfields = 1;

  // Write time in the VTK file
  ssheader << "FIELD FieldData " << nfields << std::endl;

  ssheader << "TIME 1 1 float" << std::endl;
  // Flush the ascii header
  header = ssheader.str();
  WriteHeaderString(header.c_str(), fileHdl);
  // reset the string stream
  ssheader.str(std::string());

  // convert time to single precision big endian
  float timeBE = bigEndian(static_cast<float>(time));

  WriteHeaderBinary(&timeBE, 1, fileHdl);
  // Done, add cariage return for next ascii write
  ssheader << std::endl;

  ssheader << "DIMENSIONS " << nx1 << " " << nx2 << " " << nx3 
           << std::endl;
  ssheader << "ORIGIN " << grid->xbeg_glob[IDIR] << " " 
                        << grid->xbeg_glob[JDIR] << " " 
                        << grid->xbeg_glob[KDIR] 
                        << std::endl;
  ssheader << "SPACING " << grid->dx[IDIR] << " " 
                         << grid->dx[JDIR] << " " 
                         << grid->dx[KDIR] 
                         << std::endl;

  header = ssheader.str();
  WriteHeaderString(header.c_str(), fileHdl);

  /* -----------------------------------------------------
  5. Dataset attributes [will continue by later calls
    to WriteVTK_Vector() or WriteVTK_Scalar()...]
  ----------------------------------------------------- */
  // reset the string stream
  ssheader.str(std::string());
  ssheader << std::endl << "POINT_DATA " << nx1 * nx2 * nx3 << std::endl;
  header = ssheader.str();
  WriteHeaderString(header.c_str(), fileHdl);
}

void Vtk::WriteHeaderBinary(float *buffer, int64_t nelem, VtkFileHandler fvtk) {
  #ifdef WITH_MPI
  
    MPI_Status status;
    MPI_File_set_view(fvtk, this->offset, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL );
    if(this->isRoot) {
      MPI_File_write(fvtk, buffer, nelem*sizeof(float), MPI_BYTE, &status);
    }
    offset=offset+nelem*sizeof(float);
    
  #else
    if(fwrite(buffer, sizeof(float), nelem, fvtk) != nelem) {
      throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
    }
  #endif
  }

void Vtk::WriteHeaderString(const char* header, VtkFileHandler fvtk) {
  #ifdef WITH_MPI
  
    MPI_Status status;
    MPI_File_set_view(fvtk, this->offset, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL );
    if(this->isRoot) {
      MPI_File_write(fvtk, header, strlen(header), MPI_CHAR, &status);
    }
    offset=offset+strlen(header);
  #else
    int rc = fprintf (fvtk, "%s", header);
    if(rc<0) {
      throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
    }
  #endif
  }



Vtk::~Vtk() {
  
#ifdef WITH_MPI
  MPI_Type_free(&this->view);
  MPI_File_close(&fileHdl);
#else
  fclose(fileHdl);
#endif
  delete this->vect3D;
  // Make file number
  astra::cout << "done in " << timer.seconds() << " s." << std::endl;
}


/* ********************************************************************* */
void Vtk::WriteScalar(VtkFileHandler fvtk, float* Vin,  const std::string &var_name) {
/*!
* Write VTK scalar field.
*
*********************************************************************** */

  std::stringstream ssheader;

  ssheader << std::endl << "SCALARS " << var_name.c_str() << " float" << std::endl;
  ssheader << "LOOKUP_TABLE default" << std::endl;
  std::string header(ssheader.str());

  WriteHeaderString(header.c_str(), fvtk);

#ifdef WITH_MPI
  MPI_File_set_view(fvtk, this->offset, MPI_FLOAT, this->view,
                                  "native", MPI_INFO_NULL);

  int nwrite = nx1loc*nx2loc*nx3loc;

  MPI_File_write_all(fvtk, Vin, nwrite, MPI_FLOAT, MPI_STATUS_IGNORE);

  this->offset = this->offset + sizeof(float)*nx1*nx2*nx3;
  
#else
  if(fwrite(Vin,sizeof(float),nx1loc*nx2loc*nx3loc,fvtk) != nx1loc*nx2loc*nx3loc) {
    throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
  }
#endif
}