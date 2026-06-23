Quickstart
==========

This page provides a minimal workflow to build and run **Astra**.

What is Astra?
--------------

**Astra** (Accelerated Spectral code for TuRbulent plAsmas) is a modern,
high-performance spectral plasma turbulence code. It is designed for
performance portability with `Kokkos <https://github.com/kokkos/kokkos>`_
and FFT backends through
`Kokkos-fft <https://github.com/kokkos/kokkos-fft>`_.

Supported backends include:

* **CPU** (OpenMP/Threads + FFTW)
* **NVIDIA GPU** (CUDA + cuFFT)
* **AMD GPU** (HIP/ROCm + rocFFT)

Prerequisites
-------------

You need:

* CMake >= 3.16
* A C++20-capable compiler
* Backend-specific FFT library:

  * CPU: FFTW3
  * NVIDIA: CUDA Toolkit (cuFFT)
  * AMD: ROCm (rocFFT)

* Optional: MPI implementation (OpenMPI, MPICH, ...)

Download Astra
--------------

You can get the latest source code from `GitHub <https://github.com/glesur/astra>`_ by cloning the repository with:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/glesur/astra.git astra
    cd astra


.. note::

    If you are updating an already cloned Astra installation (e.g. ``git pull``), you can use the ``git submodule update --init --recursive`` command to automatically update submodules.

Build Astra
-----------

It is recommended to build in a dedicated directory.

1. Create and enter a build directory:

    .. code-block:: bash

        mkdir build
        cd build

2. Configure with CMake for your target backend.

    **CPU (FFTW):**

    .. code-block:: bash

        cmake .. -DAstra_MPI=ON

    **NVIDIA GPU (CUDA + cuFFT):**

    .. code-block:: bash

        cmake .. \
             -DKokkos_ENABLE_CUDA=ON \
             -DKokkos_ARCH_VOLTA70=ON \
             -DAstra_MPI=ON

    **AMD GPU (ROCm + rocFFT):**

    .. code-block:: bash

        cmake .. \
             -DKokkos_ENABLE_HIP=ON \
             -DAstra_MPI=ON

3. Compile:

    .. code-block:: bash

        make -j

Run Astra
---------

Run in serial:

.. code-block:: bash

    ./astra -i ../problem/hydro_turbulence.ini

Run with MPI (example with 4 ranks):

.. code-block:: bash

    mpirun -np 4 ./astra -i ../problem/hydro_turbulence.ini

Next step
---------

After this first run, explore available problem setup files in ``problem/``
and adapt input parameters in ``.ini`` files for your target simulations.
