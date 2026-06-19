Command line & Signal handling
==============================

.. _commandLine:

Command line options
--------------------

Several options can be provided at command line when running the code. These are listed below

+--------------------+-------------------------------------------------------------------------------------------------------------------------+
| Option name        | Comment                                                                                                                 |
+====================+=========================================================================================================================+
| -restart n         | | Restart from the ``n``^th dump file. By default, ``n`` matches the highest value from existing dump files.            |
|                    | | When used, the initial conditions  are ignored.                                                                       |
+--------------------+-------------------------------------------------------------------------------------------------------------------------+
| -i                 |   specify the name of the input file to be used (default ``astra.ini``)                                                 |
+--------------------+-------------------------------------------------------------------------------------------------------------------------+
| -maxcycles n       |   stops when the code has performed ``n`` integration cycles                                                            |
+--------------------+-------------------------------------------------------------------------------------------------------------------------+
| -nolog             |   disable log files                                                                                                     |
+--------------------+-------------------------------------------------------------------------------------------------------------------------+
| -nowrite           |   disable all writes (useful for raw performance measures or for tests). This option implies ``-nolog``                 |
+--------------------+-------------------------------------------------------------------------------------------------------------------------+
| -profile           |   Enable on-the-fly performance profiling (a final text report is automatically generated).                             |
+--------------------+-------------------------------------------------------------------------------------------------------------------------+
| -test              |   Run integrated unit tests.                                                                                            |
+--------------------+-------------------------------------------------------------------------------------------------------------------------+




.. _signalHandling:

Signal Handling
---------------

By default, *Astra* is designed to capture the ``SIGUSR2`` UNIX signal sent by the host operating system when it is running. When such a signal is captured, *Astra* finishes
its current integration step, dumps a restart file and ends. This signal handling is therefore useful to tell *Astra* that its allocated time for the current
job is ending, and it is therefore time to stop the integration. Many modern jobs schedulers, such as OAR and SLURM, allow users to send UNIX signals
before killing the jobs. Read the documentation of your job scheduler for more information.

If you are not using a scheduler, it is possible to interupt *Astra* by directly sending it a ``SIGUSR2`` UNIX
signal with the ``kill`` command. For instance:

.. code-block:: bash

  kill -s SIGUSR2  123456

where 123456 is *Astra* pid (obtainable with ``ps -a``). When runnning with MPI, the pid used in the ``kill`` command can be any *Astra* process, as *Astra*
automatically broadcast the abort signal to all of its running processes.

Command file
------------

In cases when *Astra* processes cannot be reached directly (as in the case when the executable runs on compute nodes on which it is not possible to connect),
it is still possible to interupt *Astra* by creating an empty file with a specific filename (i.e. a "command file") in the directory where *Astra* is running (for instance
using the command ``touch``). Currently, *Astra* only supports the command file ``stop``. Hence

.. code-block:: bash

  touch stop

has exactly the same effect as sending the ``SIGUSR2`` signal to *Astra*: it triggers the creation of a restart dump and finishes.
