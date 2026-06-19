Code configuration with Cmake
=============================


Introduction to Cmake
+++++++++++++++++++++
`Cmake <https://cmake.org>`_ is a tool to control code generation on diverse platforms. It is the default tool used to configure *Astra*. *Astra* (and Kokkos)
requires ``cmake`` version >= 3.16. It is also recommended to use the graphical frontend ``ccmake`` to configure *Astra*, as it allows one to have a rapid
overview of all of the configuration options and switch them according to the target architecture.

To configure *Astra* with ``Cmake``, simply launch ``cmake <Astra_DIR>`` with the desired options where `<Astra_DIR>` is the path to the *Astra* source directory. It is recommended to create a separate build directory (e.g. ``build``) and launch ``cmake`` from there, as in the following example:

.. code-block:: bash

  mkdir build
  cd build
  cmake ..

Alternatively, you can replace ``cmake`` by ``ccmake`` to get a more user-friendly graphical interface.


.. _configurationOptions:

Main configuration options
++++++++++++++++++++++++++

Several options can be enabled from the command line (or are accessible with ``ccmake`` GUI):

``-LH``
    List all of the available configure options and exit

``-D Astra_MPI=ON``
    Enable MPI parallelisation. Requires an MPI library. When used in conjonction with CUDA (Nvidia GPUs), a CUDA-aware MPI library is required by *Astra*.


``-D Astra_DEBUG=ON``
    Enable debug options in *Astra*. This triggers a lot of outputs, and automatic bound checks of array accesses. As a result, this
    option makes the code very slow.

``-D Kokkos_ENABLE_OPENMP=ON``
    Enable OpenMP parallelisation on supported compilers. Note that this can be enabled simultaneously with MPI, resulting in a hybrid MPI+OpenMP compilation.

``-D Kokkos_ENABLE_CUDA=ON``
    Enable Nvidia Cuda (for GPU targets). When enabled, ``cmake`` will attempt to auto-detect the target GPU architecture. If this fails, one needs to specify
    the target architecture adding ``-DKokkos_ARCH_{..}=ON`` (see below).

``-D Kokkos_ARCH_{...}=ON``
    Enable architecture-specific optimisation. A complete list can be obtained with the ``-LH`` option. Note that several host and target architecture can be enabled
    simulatenously (e.g for a CPU and a GPU). For instance:

      + Intel CPUs: BDW (Broadwell), HSW (Haswell), KNL (Knights Landing Xeon phi), SKX (Skylake Xeon with AVX512), SNB (Sandy/Ivy bridge), WSM (Westmere) ...
      + NVIDIA GPUs: PASCAL60, PASCAL61, VOLTA70, VOLTA72, AMPERE80, AMPERE86, ...
      + IBM CPUs: POWER7, POWER8, POWER9, BOOL (Blue gene Q)
      + ARM CPUs: ARMV80, ARMV81, ARMV8_THUNDERX, ARMV8_THUNDERX2, A64FX...
      + AMD CPUs: AMDAVX, ZEN, ZEN2, ZEN3...
      + AMD GPUs: VEGA906, VEGA908...


``-D CMAKE_CXX_COMPILER=foo``
    Request a specific ``foo`` compiler for the compilation and link. Alternatively, it is also possible to export the ``CXX`` environement variable with a valid compiler name
    before calling ``cmake``.

.. tip::

    Note that ``cmake`` keeps a cache of the previous configuration performed in a particular build directory. To reset the configuration and start from scratch,
    delete the file `CMakeCache.txt` or use the ``--fresh`` option.


.. _setupExamples:

Configuration examples for selected clusters
++++++++++++++++++++++++++++++++++++++++++++


AdAstra at CINES, AMD Mi250X GPUs
---------------------------------

We recommend the following modules and environement variables on AdAstra:

.. code-block:: bash

    module load cpe/24.07
    module load craype-accel-amd-gfx90a craype-x86-trento
    module load PrgEnv-cray
    module load amd-mixed/6.1.2
    module load rocm/6.1.2
    module load cray-python/3.11.7

Finally, *Astra* can be configured to run on Mi250 by enabling HIP and the desired architecture with the following options to ccmake:

.. code-block:: bash

    -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON -DKokkos_ARCH_VEGA90A=ON


MPI (multi-GPU) can be enabled by adding ``-DAstra_MPI=ON`` as usual.

Jean Zay at IDRIS, Nvidia V100/A100/H100 GPUs
---------------------------------------------

We recommend the following modules and environement variables on Jean Zay V100/A100:

.. code-block:: bash

    module load arch/a100 # ONLY forA100
    module load cuda/12.1.0
    module load gcc/12.2.0
    module load openmpi/4.1.1-cuda
    module load cmake/3.25.2

While for H100:

.. code-block:: bash

    module load arch/h100
    module load cmake/3.30.1
    module load cuda/12.1.0
    module load openmpi/4.1.5-cuda

*Astra* can then be configured to run on Nvidia V100 with the following options to ccmake:

.. code-block:: bash

    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON

While Ampere A100 GPUs are enabled with

.. code-block:: bash

    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON

And for H100 GPUS:

.. code-block:: bash

    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON


MPI (multi-GPU) can be enabled by adding ``-DAstra_MPI=ON`` as usual.
