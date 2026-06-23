
[![Documentation status](https://github.com/glesur/astra/actions/workflows/astra-ci-doc.yml/badge.svg?branch=master)](https://github.com/glesur/astra/actions/workflows/astra-ci-doc.yml)
[![Astra CIs](https://github.com/glesur/astra/actions/workflows/astra-ci.yml/badge.svg?branch=master)](https://github.com/glesur/astra/actions/workflows/astra-ci.yml)
# Astra

**A**ccelerated **S**pectral code for **T**uRbulent pl**A**smas

## Overview

**Astra** is a modern, high-performance spectral code designed for the simulation of turbulent plasmas. It serves as an accelerated and modernized evolution of the [Snoopy code](https://ipag.osug.fr/~lesurg/snoopy), using an input file format and internal structure similar to the [Idefix code](https://github.com/idefix-code/idefix).

By leveraging modern C++ abstraction layers, Astra achieves performance portability across a wide range of high-performance computing architectures without compromising the numerical accuracy established by its predecessor.

## Documentation

The full documentation is automatically generated and available on [GitHub Pages](https://glesur.github.io/astra/).
## Key Features

  * **Performance Portability:** Built on top of **Kokkos**, allowing a single codebase to run efficiently on multicore CPUs and various GPU architectures.
  * **Spectral Solver:** Utilizes **Kokkos-fft** to handle Fast Fourier Transforms across different hardware backends.
  * **Parallelization:** Supports distributed memory parallelism via **MPI** utilizing a 1D slab decomposition strategy. No MPI support is required from the FFT library.

## Hardware Backend Support

Thanks to the integration of [Kokkos](https://github.com/kokkos/kokkos) and [Kokkos-fft](https://github.com/kokkos/kokkos-fft), Astra supports the following hardware configurations:

  * **CPU:** Parallel execution via OpenMP/Threads with FFTs handled by **FFTW**.
  * **Nvidia GPUs:** Accelerated execution via CUDA with FFTs handled by **cuFFT**.
  * **AMD GPUs:** Accelerated execution via HIP/ROCm with FFTs handled by **rocFFT**.

## Dependencies

  * CMake (3.16+)
  * C++ Compiler (supporting C++20 or higher)
  * **Backend specific libraries:**
      * *CPU:* FFTW3
      * *Nvidia:* CUDA Toolkit (cuFFT)
      * *AMD:* ROCm (rocFFT)
  * Optional: GPU-aware MPI Implementation (OpenMPI, MPICH, etc.)


## Download Instructions

It is recommended to directly clone Astra and its submodules (Kokkos and Kokkos-fft) from GitHub. In the directory of your choice:

    ```bash
    git clone --recurse-submodules https://github.com/glesur/astra.git astra
    cd astra
    ```


## Build Instructions

Astra utilizes CMake for its build system. It is recommended to configure the build in a dedicated directory to keep the source tree clean.

### Basic Configuration Steps

1.  Create a build directory:

    ```bash
    mkdir build
    cd build
    ```

2.  Configure the project using CMake. Select the command below that matches your target hardware. Note that MPI is optional, and the associated option can be omitted to run in serial.

    **Option A: CPU (FFTW)**

    ```bash
    cmake .. -DAstra_MPI=ON
    ```

    **Option B: Nvidia GPU (CUDA + cuFFT)**

    ```bash
    cmake .. \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_VOLTA70=ON \  # Change architecture flag as needed (e.g., AMPERE80)
        -DAstra_MPI=ON
    ```

    **Option C: AMD GPU (ROCm + rocFFT)**

    ```bash
    cmake .. \
        -DKokkos_ENABLE_HIP=ON \
        -DAstra_MPI=ON
    ```

3.  Compile the code:

    ```bash
    make -j
    ```

## Running Astra

Once compiled, the executable can be run directly (without mpi) or using standard MPI launchers (mpirun). For example, to run on 4 processes:

```bash
mpirun -np 4 ./astra -i ../problem/hydro_turbulence.ini
```

## License

[CeCILL 2.1](https://en.wikipedia.org/wiki/CeCILL)

## Contact & Attribution

For information regarding the original algorithm and physics, please refer to the original [Snoopy code website](https://ipag.osug.fr/~lesurg/snoopy).
