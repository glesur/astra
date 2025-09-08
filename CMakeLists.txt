cmake_minimum_required(VERSION 3.16)
set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 17)

project (Astra VERSION 1.0.0)

set(Astra_LOOP_PATTERN "Default" CACHE STRING "Loop pattern for astra_for")

#astra requires Cuda Lambdas (experimental)
if(Kokkos_ENABLE_CUDA)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Astra requires lambdas on Cuda" FORCE)
endif()

# Add kokkos CMAKE files (required early since these set compiler options)
add_subdirectory(src/kokkos build/kokkos)
add_subdirectory(src/kokkos-fft)


add_executable(astra)
add_subdirectory(src build)

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

target_include_directories(astra PUBLIC
                           src/kokkos/core/src
                           src
                           src/utils
                           )

target_link_libraries(astra Kokkos::kokkos KokkosFFT::fft)