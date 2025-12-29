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

Dump::Dump(Grid *grid, std::string filebase, std::string directory) {

  
  // Initialise the root tag (used for MPI non-collective I/Os)
  this->isRoot = astra::prank == 0;
  this->grid_nglob = grid->npf_glob[0]*grid->npf_glob[1]*grid->npf_glob[2];
  this->filebase = filebase;
  this->directory = directory;

  // Create MPI view when using MPI I/O
  #ifdef WITH_MPI
    int start[3];
    int size[3];
    int subsize[3];

    for(int dir = 0; dir < 3 ; dir++) {
      // VTK assumes Fortran array ordering, hence arrays dimensions are filled backwards
      start[2-dir] = 0;
      size[2-dir] = grid->npf_glob[dir];
      subsize[2-dir] = grid->npf[dir];
    }
    // Fix starting point along x
    start[2] = grid->npf[0]*astra::prank;
    
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
  fs::path outputDirectory = directory;
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
  fs::path filename;
  std::stringstream ssfileName;
  ssfileName << filebase << ".dmp";
  filename = outputDirectory/ssfileName.str();

  astra::cout << "Dump: Opening file " << ssfileName.str() << "..." << std::flush;
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

void Dump::Read() {
  astra::pushRegion("Dump::Read");
  DumpFileHandler fileHdl;
  // Open file;
  fs::path filename;
  fs::path outputDirectory = directory;
  std::stringstream ssfileName;
  ssfileName << filebase << ".dmp";
  
  filename = outputDirectory/ssfileName.str();

  astra::cout << "Dump: Reading file " << ssfileName.str() << "..." << std::flush;
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
      msg << "Failed to open dump file: " << std::string(filename) << std::endl;
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
      ArrayHost3D<complex> arrHost("temp", dim[0], dim[1], dim[2]);
      ReadData(fileHdl, arrHost);
      // Check if Field name exists
      if(fields.find(fieldName) == fields.end()) {
        //  Create field
        fields[fieldName] = Field<Array3D<complex>>(fieldName,{dim[0], dim[1], dim[2]});
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

