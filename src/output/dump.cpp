// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include "dump.hpp"
#include "grid.hpp"
#include "global.hpp"
#include "version.hpp"
#if __has_include(<filesystem>)
  #include <filesystem> // NOLINT [build/c++17]
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

Dump::Dump(Grid *grid, std::string filename) {

  
  // Initialise the root tag (used for MPI non-collective I/Os)
  this->isRoot = astra::prank == 0;
  this->npf = grid->npf;
  this->npf_glob = grid->npf_glob;
  this->npr_glob= grid->npr_glob;

  this->filename = filename;

  // Create MPI view when using MPI I/O
  #ifdef WITH_MPI
    int start[3];
    int size[3];
    int subsize[3];

    for(int dir = 0; dir < 3 ; dir++) {
      // VTK assumes Fortran array ordering, hence arrays dimensions are filled backwards
      start[dir] = 0;
      size[dir] = grid->npf_glob[dir];
      subsize[dir] = grid->npf[dir];
    }
    // Fix starting point along x
    start[0] = grid->npf[0]*astra::prank;
    
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, MPI_C_DOUBLE_COMPLEX, &this->view);
    MPI_Type_commit(&this->view);
  #endif // WITH_MPI
}

void Dump::Register(std::string name, Field<Array3D<complex>> field) {
  fields[name] = field;
}

void Dump::Register(std::string name, real value) {
  reals[name] = value;
}

void Dump::Register(std::string name, int value) {
  ints[name] = value;
}

void Dump::Write() {
  astra::pushRegion("Dump::Write");
  fs::path outputDirectory = this->filename.parent_path();
  DumpFileHandler fileHdl;

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

  astra::cout << "Dump: Opening file " << this->filename.filename() << "..." << std::flush;
  // Reset timer
  Kokkos::Timer timer;
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
                  MPI_INFO_NULL, &fileHdl);
    this->offset = 0;
  #else
    fileHdl = fopen(filename.c_str(),"wb");

    if(fileHdl == NULL) {
      std::stringstream msg;
      msg << "Unable to open file " << filename << std::endl;
      msg << "Check that you have write access and that you don't exceed your quota." << std::endl;
      throw std::runtime_error(msg.str());
    }
  #endif

  // Header
  WriteString(fileHdl, "ASTRA "+std::string(ASTRA_VERSION)+ " Dump file");

  for(auto& field : fields) {
    for(auto& array: field.second) {
      ArrayHost3D<complex> arrHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), array.second);
      WriteData(fileHdl, field.first+"\\"+array.first, arrHost);
    }
  }
  for(auto& real : reals) {
    WriteData(fileHdl, real.first, real.second);
  }
  for(auto& integer : ints) {
    WriteData(fileHdl, integer.first, integer.second);
  }
  int endTag = -9999;
  WriteData(fileHdl, "EOF", endTag);
  #ifdef WITH_MPI
    MPI_File_close(&fileHdl);
  #else
    fclose(fileHdl);
  #endif
  astra::cout << "done in " << timer.seconds() << " s." << std::endl;
  astra::popRegion();
}

void Dump::ReadSnoopy() {
astra::pushRegion("Dump::ReadSnoopy");
  DumpFileHandler fileHdl;


  astra::cout << "Dump: Reading Snoopy's dump " << this->filename.c_str() << "..." << std::flush;
  // Reset timer
  Kokkos::Timer timer;
  timer.reset();
  // Open file
  #ifdef WITH_MPI
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                              MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN,
                              MPI_INFO_NULL, &fileHdl);
    this->offset = 0;
  #else
    fileHdl = fopen(filename.c_str(),"rb");
    if(fileHdl == NULL) {
      std::stringstream msg;
      msg << "Failed to open dump file: " << this->filename.filename() << std::endl;
      throw std::runtime_error(msg.str());
    }
  #endif


  // Read and check header
  int dumpVersion;
  ReadSnoopyScalar(fileHdl, dumpVersion); // Ignore version for now
  astra::cout << "Snoopy dump (version " << dumpVersion << ") " << std::endl;
  int nx1, nx2, nx3;
  ReadSnoopyScalar(fileHdl, nx1);
  ReadSnoopyScalar(fileHdl, nx2);
  ReadSnoopyScalar(fileHdl, nx3);
  astra::cout << "with grid size (" << nx1 << ", " << nx2 << ", " << nx3 << ") " << std::endl;
  int included_fields;
  ReadSnoopyScalar(fileHdl, included_fields);
  astra::cout << included_fields << " fields included." << std::endl;
  
  // Check that grid size matches
  if(nx1 != npr_glob[0] || nx2 != npr_glob[1] || nx3 != npr_glob[2]) {
    throw std::runtime_error("Error: grid size in dump file (" 
          + std::to_string(nx1) + ", " + std::to_string(nx2) + ", " + std::to_string(nx3) +
           ") does not match simulation grid size.");
  }
  fields["state"] = Field<Array3D<complex>>("state",npf);
  ReadSnoopyArray(fileHdl, "vx1", fields["state"]);
  ReadSnoopyArray(fileHdl, "vx2", fields["state"]);
  ReadSnoopyArray(fileHdl, "vx3", fields["state"]);

  if(included_fields & 1) {
    // Boussinesq field
    ReadSnoopyArray(fileHdl, "th", fields["state"]);
  }
  if(included_fields & 16) {
    // concentration field
    ReadSnoopyArray(fileHdl, "c", fields["state"]);
  }
  if(included_fields & 32) {
    // concentration field
    ReadSnoopyArray(fileHdl, "ch", fields["state"]);
  }
  if(included_fields & 2) {
    // Magnetic field
    ReadSnoopyArray(fileHdl, "bx1", fields["state"]);
    ReadSnoopyArray(fileHdl, "bx2", fields["state"]);
    ReadSnoopyArray(fileHdl, "bx3", fields["state"]);
  }
  if(included_fields & 4) {
    // Particles
    throw std::runtime_error("Error: reading particles from Snoopy dump files is not supported yet.");
  }
  if(included_fields & 8) {
    // Compressible density field
    ReadSnoopyArray(fileHdl, "rho", fields["state"]);
  }
  ReadSnoopyScalar(fileHdl,reals["time"]);
  ReadSnoopyScalar(fileHdl,ints["nvtk"]);
  ReadSnoopyScalar(fileHdl,ints["slice"]);
  ReadSnoopyScalar(fileHdl,ints["ndmp"]);
  ReadSnoopyScalar(fileHdl,reals["lastTimevar"]);
  ReadSnoopyScalar(fileHdl,reals["lastVtkOutput"]);
  ReadSnoopyScalar(fileHdl,reals["lastDmpOutput"]);
  ReadSnoopyScalar(fileHdl,reals["lastSliceOutput"]);

  int marker;
  ReadSnoopyScalar(fileHdl,marker); // Read end marker
  if(marker != 1981) {
    throw std::runtime_error("Error: invalid end of Snoopy dump file.");
  }
  #ifdef WITH_MPI
  MPI_File_close(&fileHdl);
  #else
  fclose(fileHdl);
  #endif

  astra::cout << "done in " << timer.seconds() << " s." << std::endl;
  astra::popRegion();
}


void Dump::Read() {
  astra::pushRegion("Dump::Read");
  DumpFileHandler fileHdl;
  // Open file;

  astra::cout << "Dump: Reading file " << this->filename.c_str() << "..." << std::flush;
  // Reset timer
  Kokkos::Timer timer;
  timer.reset();
  // Open file
  #ifdef WITH_MPI
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                              MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN,
                              MPI_INFO_NULL, &fileHdl);
    this->offset = 0;
  #else
    fileHdl = fopen(this->filename.c_str(),"rb");
    if(fileHdl == NULL) {
      std::stringstream msg;
      msg << "Failed to open dump file: " << std::string(this->filename) << std::endl;
      throw std::runtime_error(msg.str());
    }
  #endif
  
  // Read and check header
  std::string header = ReadString(fileHdl);
  if(header.substr(0,6)!="ASTRA ") {
    throw std::runtime_error("Invalid dump file: "+std::string(filename));
  }

  while(true) {
    std::vector<int> dim;
    DataType type;
    std::string name;
    ReadNextFieldProperties(fileHdl, dim, type, name);
    if(name.compare("EOF") == 0) {
      break;
    }

    if(type == ComplexDoubleType && dim.size() == 3) {
      std::string fieldName = name.substr(0,name.find("\\"));
      std::string arrayName = name.substr(name.find("\\")+1);
      ArrayHost3D<complex> arrHost("temp", npf[0], npf[1], npf[2]);
      ReadData(fileHdl, arrHost);
      // Check if Field name exists
      if(fields.find(fieldName) == fields.end()) {
        //  Create field
        fields[fieldName] = Field<Array3D<complex>>(fieldName,npf);
      }
      fields[fieldName].Add(arrayName);
      Kokkos::deep_copy(fields[fieldName][arrayName],arrHost);;
    } else if(type == DoubleType) {
      real &value = reals[name];
      ReadData(fileHdl, value);
    } else if(type == IntegerType) {
      int &value = ints[name];
      ReadData(fileHdl, value);
    } else {
      throw std::runtime_error("Unsupported data type in dump file for variable "+name);
    }
  }
  #ifdef WITH_MPI
  MPI_File_close(&fileHdl);
  #else
  fclose(fileHdl);
  #endif

  astra::cout << "done in " << timer.seconds() << " s." << std::endl;
  astra::popRegion();
}

void Dump::Fetch(std::string name, Field<Array3D<complex>> &field) {
  field.CopyFrom(this->fields.at(name));
}

void Dump::Fetch(std::string name, real &value) {
  value = this->reals.at(name);
}

void Dump::Fetch(std::string name, int &value) {
  value = this->ints.at(name);
}

// Internal functions
///////////////////////////////////
// Write a string
void Dump::WriteString(DumpFileHandler fileHdl, std::string string) {
  astra::pushRegion("Dump::WriteString");
  char str[StringMaxLength];
  std::strncpy(str, string.c_str(), StringMaxLength-1);
  // Add terminal null character, just in case
  str[StringMaxLength-1]='\0';

  #ifdef WITH_MPI
    MPI_Status status;
    MPI_File_set_view(fileHdl, this->offset,
                                    MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL );
    if(isRoot) {
      MPI_File_write(fileHdl, str, StringMaxLength, MPI_CHAR, &status);
    }
    offset=offset+StringMaxLength;
  #else
    if(fwrite (str, sizeof(char), StringMaxLength, fileHdl) != StringMaxLength) {
      throw std::runtime_error("Unable to write to file. Check your filesystem permissions and disk quota.");
    }
  #endif
  astra::popRegion();
}

std::string Dump::ReadString(DumpFileHandler fileHdl) {
  astra::pushRegion("Dump::ReadString");
  char str[StringMaxLength];
  #ifdef WITH_MPI
    MPI_Status status;
    MPI_File_set_view(fileHdl, this->offset,
                                    MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL );
    if(isRoot) {
      MPI_File_read(fileHdl, str, StringMaxLength, MPI_CHAR, &status);
    }
    offset=offset+StringMaxLength;
    // Broadcast
    MPI_Bcast(str, StringMaxLength, MPI_CHAR, 0, MPI_COMM_WORLD);
  #else
    size_t numRead = fread(str, sizeof(char), StringMaxLength, fileHdl);
    if(numRead<StringMaxLength) {
      throw std::runtime_error("Error: unexpected end of dump file");
    }
  #endif
  astra::popRegion();
  return std::string(str);
}


void Dump::ReadNextFieldProperties(DumpFileHandler fileHdl, std::vector<int> &dim,
                                         DataType &type, std::string &name) {
  name = ReadString(fileHdl);

  #ifdef WITH_MPI
    MPI_Status status;
    int ndim;
    // Read Datatype
    MPI_File_set_view(fileHdl, this->offset, MPI_BYTE,
                                    MPI_CHAR, "native", MPI_INFO_NULL );
    if(isRoot) {
      MPI_File_read(fileHdl, &type, 1, MPI_INT, &status);
    }
    offset=offset+sizeof(int);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Read Dimensions
    MPI_File_set_view(fileHdl, this->offset, MPI_BYTE,
                                    MPI_CHAR, "native", MPI_INFO_NULL );
    if(isRoot) {
      MPI_File_read(fileHdl, &ndim, 1, MPI_INT, &status);
    }
    offset=offset+sizeof(int);
    MPI_Bcast(&ndim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_File_set_view(fileHdl, this->offset, MPI_BYTE,
                                    MPI_CHAR, "native", MPI_INFO_NULL );
    int *dimArray = new int[ndim];
    if(isRoot) {
      MPI_File_read(fileHdl, dimArray, ndim, MPI_INT, &status);
    }
    offset=offset+sizeof(int)*ndim;
    MPI_Bcast(dimArray, ndim, MPI_INT, 0, MPI_COMM_WORLD);
    dim.assign(dimArray, dimArray+ndim);
    delete[] dimArray;

  #else
    size_t numRead;

    // Read datatype
    numRead = fread(&type, sizeof(int), 1, fileHdl);
    if(numRead<1) {
      throw std::runtime_error("Error: unexpected end of dump file");
    }
    
    int ndim;
    // read dimensions
    numRead = fread(&ndim, sizeof(int), 1, fileHdl);
    if(numRead<1) {
      throw std::runtime_error("Error: unexpected end of dump file");
    }

    int *dimArray = new int[ndim];
    numRead = fread(dimArray, sizeof(int), ndim, fileHdl);
    if(numRead<ndim) {
      throw std::runtime_error("Error: unexpected end of dump file");
    }
    dim.assign(dimArray, dimArray+ndim);
    delete[] dimArray;
  #endif
}

void Dump::ReadSnoopyArray(DumpFileHandler fileHdl, std::string name, Field<Array3D<complex>> &data) {
  astra::pushRegion("Dump::ReadSnoopyArray");
  int ntot,size,nglob;
  ntot = data.GetDimensions()[0]*data.GetDimensions()[1]*data.GetDimensions()[2];
  size = sizeof(complex);
  nglob = npf_glob[0]*npf_glob[1]*npf_glob[2];
  
  ArrayHost3D<complex> temp("temp", data.GetDimensions()[0], data.GetDimensions()[1], data.GetDimensions()[2]);
  #ifdef WITH_MPI
    MPI_Status status;

    MPI_File_set_view(fileHdl, offset, MPI_C_DOUBLE_COMPLEX, this->view, "native", MPI_INFO_NULL );
    MPI_File_read_all(fileHdl, temp.data(), ntot, MPI_C_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);


    offset+= nglob*size;
  #else
    if(fread(temp.data(), size, ntot, fileHdl) < ntot) {
      throw std::runtime_error("Error: unexpected end of dump file");
    }
  #endif

  // Allocate data and copy to the field
  data.Add(name);
  Kokkos::deep_copy(data[name], temp);
  astra::popRegion();
}