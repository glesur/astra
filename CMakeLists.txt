cmake_minimum_required(VERSION 3.16)
set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 17)

set(Astra_VERSION_MAJOR 0)
set(Astra_VERSION_MINOR 1)
set(Astra_VERSION_PATCH 00)

project (Astra VERSION 0.1.00)

set(Astra_LOOP_PATTERN "Default" CACHE STRING "Loop pattern for astra_for")
option(Astra_DEBUG "Enable Astra debug features (makes the code very slow)" OFF)
option(Astra_MPI "enable Message Passing Interface parallelisation" OFF)

# load git revision tools
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescription)


#astra requires Cuda Lambdas (experimental)
if(Kokkos_ENABLE_CUDA)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Astra requires lambdas on Cuda" FORCE)
endif()

# Add kokkos CMAKE files (required early since these set compiler options)
add_subdirectory(src/kokkos build/kokkos)
add_subdirectory(src/kokkos-fft)


add_executable(astra)
add_subdirectory(src build)

#update version.hpp if possible
git_describe(GIT_SHA1)
set(Astra_VERSION ${Astra_VERSION_MAJOR}.${Astra_VERSION_MINOR}.${Astra_VERSION_PATCH}-${GIT_SHA1})
file(WRITE ${CMAKE_SOURCE_DIR}/src/version.hpp "#define ASTRA_GIT_COMMIT \"${GIT_SHA1}\"\n")
file(APPEND ${CMAKE_SOURCE_DIR}/src/version.hpp "#define ASTRA_VERSION_MAJOR \"${Astra_VERSION_MAJOR}\"\n")
file(APPEND ${CMAKE_SOURCE_DIR}/src/version.hpp "#define ASTRA_VERSION_MINOR \"${Astra_VERSION_MINOR}\"\n")
file(APPEND ${CMAKE_SOURCE_DIR}/src/version.hpp "#define ASTRA_VERSION_PATCH \"${Astra_VERSION_PATCH}\"\n")
file(APPEND ${CMAKE_SOURCE_DIR}/src/version.hpp "#define ASTRA_VERSION \"${Astra_VERSION}\"\n")

#Loop type
if(${Astra_LOOP_PATTERN} STREQUAL "SIMD")
  if(Kokkos_ENABLE_OPENMP)
    message(ERROR "SIMD loop pattern is incompatible with OPENMP")
  endif()
  if(Kokkos_ENABLE_CUDA)
    message(ERROR "SIMD loop pattern is incompatible with Cuda")
  endif()
  if(Kokkos_ENABLE_HIP)
    message(ERROR "SIMD loop pattern is incompatible with HIP")
  endif()
  add_compile_definitions("LOOP_PATTERN_SIMD")
elseif(${Astra_LOOP_PATTERN} STREQUAL "Range")
  add_compile_definitions("LOOP_PATTERN_1DRANGE")
elseif(${Astra_LOOP_PATTERN} STREQUAL "MDRange")
  add_compile_definitions("LOOP_PATTERN_MDRANGE")
elseif(${Astra_LOOP_PATTERN} STREQUAL "TeamPolicy")
  add_compile_definitions("LOOP_PATTERN_TPX")
elseif(${Astra_LOOP_PATTERN} STREQUAL "TeamPolicyInnerVector")
  add_compile_definitions("LOOP_PATTERN_TPTTRTVR")
elseif(NOT ${Astra_LOOP_PATTERN} STREQUAL "Default")
  message(ERROR "Unknown loop Pattern")
endif()

# Debug mode
if(Astra_DEBUG)
  add_compile_definitions("DEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

if(Astra_MPI)
  add_compile_definitions("WITH_MPI")
  find_package(MPI REQUIRED)
  target_link_libraries(astra MPI::MPI_CXX)
endif()

target_include_directories(astra PUBLIC
                           src/kokkos/core/src
                           src
                           src/utils
                           src/initFlow
                           src/timeIntegrator
                           src/rightHandSide
                           src/output
                           src/output/timevar
                           )

target_link_libraries(astra Kokkos::kokkos KokkosFFT::fft)