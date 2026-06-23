.. Astra documentation master file, created by
   sphinx-quickstart on Mon Sep 21 10:36:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##############################
Astra code documentation
##############################
=================
About *Astra*
=================

*Astra* is designed to be a performance-portable code to model astrophysical plasmas with spectral accuracy. This means that it can run both on your laptop's cpu or on the largest GPU HPCs recently
bought by your university. More technically, *Astra* can run in serial, use OpenMP and/or MPI (message passing interface) for parallelization, and use CUDA or Xeon-Phi for
out of chip acceleration. Of course, all these capabilities are embedded within one single code, so the code relies on relatively abstracted classes and objects available in C++20, which are not necessarily
familiar to astrophysicists. A large effort has been devoted to simplify this level of abstraction so that the code can be modified by researchers and students familiar with C and who are aware of basic object-oriented concepts.

The algorithms implemented in *Astra* are based on the `Snoopy Code <https://ipag.osug.fr/~lesurg/snoopy>`_.

================
Requirements
================
*Astra* is written in standard C++20 and does not rely on any external libraries when running in serial (non-MPI mode).

Compiler
  *Astra* requires a C++20 compatible compiler. It has been tested successfully with GCC (>10), Intel compiler suite (>2018) and
  Clang on both Intel and AMD CPUs. *Astra* has also been tested on NVIDIA GPUs (Pascal, Volta and Ampere architectures) using the nvcc (>10) compiler, and on AMD GPUs (Radeon Mi50, Mi210, Mi250) using the hipcc compiler.

Kokkos and kokkos-fft libraries
  *Astra* relies internally on the `Kokkos <https://github.com/kokkos/kokkos>`_  and  `kokkos-fft <https://github.com/kokkos/kokkos-fft>`_ libraries, which are bundled with *Astra* as git submodules and compiled on the fly, hence no external installation is required.

FFT Library
  *Astra* requires an external library to compute Fast Fourier Transforms (FFT). *Astra* has been tested successfully with the `FFTW <http://www.fftw.org/>`_ library on CPU architectures, and with the `cuFFT <https://developer.nvidia.com/cufft>`_ and `rocFFT <https://rocfft.readthedocs.io/>`_ libraries on GPU architectures.
  Note that *Astra* does not rely on the MPI version of these libraries, so the serial version of the library is sufficient even when using *Astra* in MPI parallelisation mode.


MPI library
  When using MPI parallelisation, *Astra* relies on an external MPI library. *Astra* has been tested successfully with OpenMPI and IntelMPI libraries. When used on GPU architectures, *Astra* assumes that
  the MPI library is GPU-Aware. If unsure, check this last point with your system administrator.


================
Features
================
* incompressible hydrodynamics using a pseudo-spectral method in 3D
* incompressible magneto-hydrodynamics using a pseudo-spectral method in 3D



===========================
Terms and condition of Use
===========================
*Astra* is distributed freely under the `CeCILL license <https://en.wikipedia.org/wiki/CeCILL>`_, a free software license adapted to both international and French legal matters, in the spirit of and retaining
compatibility with the GNU General Public License (GPL).

*Astra* also relies on the `Kokkos <https://github.com/kokkos/kokkos>`_ performance portability programming ecosystem released under the terms
of Contract DE-NA0003525 with National Technology & Engineering Solutions of Sandia, LLC (NTESS).

==================
Main Contributors
==================

Geoffroy Lesur
  code design and architecture, implementation of the main algorithms, documentation and testing.

*Astra* has also been inspired by all of the contributors of the `Idefix code <https://github.com/idefix-code/idefix>`_, from which the general framework has been derived.

========================
About this documentation
========================

This documentation has automatically been generated on |today| from the following *Astra* commit:

.. git_commit_detail::
    :branch:
    :commit:
    :sha_length: 10
    :no_github_link:

===================
Acknowledgements
===================

Astra developement team is partly funded by the `PEPR Origins <https://pepr-origins.fr>`_ through the project "MHD@Exascale".
The Astra collaboration benefited from funding from the “Programme National de Physique Stellaire” (PNPS),
“Programme National Soleil-Terre” (PNST), “Programme National de Hautes Energies” (PNHE) and
“Programme National de Planétologie” (PNP) of CNRS/INSU co-funded by CEA and CNES.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   configuration
   commandline
   input_file
   output




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
