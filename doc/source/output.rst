.. _output:

Outputs
=======

Output formats
--------------

*Astra* uses several types of outputs you may want for your setup. By default, *Astra* allows
for 4 kinds of outputs:

* logs which essentially tell the user what *Astra* is currently doing. When running in serial, logs are sent to stdout, but when
  MPI is enabled, only the logs of the rank 0 process are sent to stdout, and each process (including rank 0) simultaneously writes a
  log file ``astra.n.log`` where *n* is the process MPI rank.
* dump files (.dmp) which are *Astra* specific binary files containing all of the data at machine precision to restart your run.
  These files are therefore the ones which are read when *Astra* is restarted.
* VTK files (.vtk) are Visualization Toolkit files, which are easily readable by visualization software such as `ParaView <https://www.paraview.org/>`_
  or `VisIt <https://wci.llnl.gov/simulation/computer-codes/visit>`_. A set of Python methods is also provided to read VTK files from your
  Python scripts in the `pytools` directory.
* timevar files are text files in CSV format that allow you to trace box-averaged quantities as a function of time. A list of possible timevar diagnostics is given in :ref:`outputSection`.
* Python scripts, defined in the input file :ref:`astraPySection`. This launches a user-defined Python function fed with Astra data. One can then directly plot or interact with Astra outputs from Python.

The output periodicity and the user-defined variables should all be declared in the input file, as described in :ref:`outputSection`.
