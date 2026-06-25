.. _inputFile:

Problem input file
=================================

The problem input file uses the extension `.ini`. By default, *Astra* uses `astra.ini` in the current directory. It is possible to start astra with an other
input file using the `-i` command line option specifying the full path to the input file.

The problem input file is read when *Astra* starts. It is split into several sections, each section name corresponding to a C++ class in *Astra* structure.
Inside each section, each line defines an entry, which can have as many parameters as one wishes.
(note that it requires at least one parameter). The input file
allows for comments, which should start with ``#``.

.. _gridSection:

``Grid`` section
--------------------
The grid section defines the grid total dimension. It consists of 3 entries ``X1-grid``, ``X2-grid`` (when DIMENSIONS>=2) and ``X3-grid`` (when DIMENSIONS=3). Each entry defines the repartition of the grid points in the corresponding direction (the grid is always rectilinear).
Each entry defines a series of grid blocks which are concatenated along the direction. Each block in a direction can have a different spacing rule (uniform, log or stretched). The definition of the Grid entries is as follows

+----------------------------+-------------------------+------------------------------+
|                            |  Allowed value          |    Example                   |
+============================+=========================+==============================+
| Entry name                 | X1/2/3-Grid             | X1-Grid                      |
+----------------------------+-------------------------+------------------------------+
| start of domain            | floating point          | 0.0                          |
+----------------------------+-------------------------+------------------------------+
| # of points in domain      | integer >= 1            | 64                           |
+----------------------------+-------------------------+------------------------------+
| end of domain              | floating point          | 1.0                          |
+----------------------------+-------------------------+------------------------------+


In the example below, we define in ``X1`` direction a grid of 64 points starting at ``X1=0.0`` and ending at ``X1=1.0``:

.. code-block::

  [Grid]
  X1-Grid        0.0    64      1.0

``Physics`` section
--------------------
The physics section defines the physics solved by *Astra* and the associated parameters.
The physics section should at least contain the entry ``rhs`` which define the right-hand side of the equations solved by *Astra*.
Depending on the choice of the right-hand side, other entries may be required.

+----------------------------+-------------------------------------------------------+
| ``rhs`` value              |  Comment                                              |
+============================+=======================================================+
| ``advection``              | An homogenous advection rhs, used for testing         |
+----------------------------+-------------------------------------------------------+
| ``hydro``                  | Incompressible Navier Stokes equations                |
+----------------------------+-------------------------------------------------------+
| ``burgers``                | 1D Burgers equations (used for testing)               |
+----------------------------+-------------------------------------------------------+
| ``mhd``                    | Incompressible magnetohydrodynamics equation          |
+----------------------------+-------------------------------------------------------+
| ``compressible_hydro``     | compressible Navier Stokes equations                  |
+----------------------------+-------------------------------------------------------+

Depending on the choice of the right-hand side, the following entries may be required in the physics section:

``advection`` right hand side
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
|  Entry name    | Parameter type     | Comment                                                                                                   |
+================+====================+===========================================================================================================+
| direction      | integer            | direction of the flow (0=x1, 1=x2, 2=x3)                                                                  |
+----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| velocity       | float              | flow velocity (constant in space and time)                                                                |
+----------------+--------------------+-----------------------------------------------------------------------------------------------------------+

``hydro`` right hand side
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
|  Entry name     | Parameter type     | Comment                                                                                                   |
+=================+====================+===========================================================================================================+
| viscosity       | float, (int)       | | 1st parameter: kinematic viscosity                                                                      |
|                 |                    | | 2nd parameter (optional): order of the viscosity term *n* in :math:`=\nu \Delta^n v (default 1)`        |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| omega           | float              | (optional) rotation rate along the x3 (=z) axis                                                           |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| shear_type      | string             | (optional) type of large-scale shear. Value allowed: ``linear``                                           |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| shear_rate      | float              | (optional) shear rate when `shear_type` is `linear`                                                       |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+



``mhd`` right hand side
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
|  Entry name     | Parameter type     | Comment                                                                                                   |
+=================+====================+===========================================================================================================+
| viscosity       | float, (int)       | | 1st parameter: kinematic viscosity                                                                      |
|                 |                    | | 2nd parameter (optional): order of the viscosity term *n* in :math:`=\nu \Delta^n v (default 1)`        |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| resistivity     | float, (int)       | | 1st parameter: kinematic resistivity                                                                    |
|                 |                    | | 2nd parameter (optional): order of the resistivity term *n* in :math:`=\nu \Delta^n B (default 1)`      |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| omega           | float              | (optional) rotation rate along the x3 (=z) axis                                                           |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| shear_type      | string             | (optional) type of large-scale shear. Value allowed: ``linear``                                           |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| shear_rate      | float              | (optional) shear rate when `shear_type` is `linear`                                                       |
+-----------------+--------------------+-----------------------------------------------------------------------------------------------------------+

``compressible_hydro`` right hand side
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


+-----------------+--------------------+------------------------------------------------------------------------------------------------------------+
|  Entry name     | Parameter type     | Comment                                                                                                    |
+=================+====================+============================================================================================================+
| viscosity       | float, (int)       | | 1st parameter: kinematic viscosity                                                                       |
|                 |                    | | 2nd parameter (optional): order of the viscosity term *n* in :math:`=\nu \Delta^n v` (default 1)         |
+-----------------+--------------------+------------------------------------------------------------------------------------------------------------+
| eta_rho         | float, (int)       | | 1st parameter: mass diffusion                                                                            |
|                 |                    | | 2nd parameter (optional): order of the diffusion term *n* in :math:`=\eta_\rho \Delta^n \rho` (default 1)|
+-----------------+--------------------+------------------------------------------------------------------------------------------------------------+
| rho_floor       | float              | (optional) density floor (default 1e-6)                                                                    |
+-----------------+--------------------+------------------------------------------------------------------------------------------------------------+
| omega           | float              | (optional) rotation rate along the x3 (=z) axis                                                            |
+-----------------+--------------------+------------------------------------------------------------------------------------------------------------+
| cs              | float              | (optional) isothermal sound speed (default 1)                                                              |
+-----------------+--------------------+------------------------------------------------------------------------------------------------------------+
| shear_type      | string             | (optional) type of large-scale shear. Value allowed: ``linear``                                            |
+-----------------+--------------------+------------------------------------------------------------------------------------------------------------+
| shear_rate      | float              | (optional) shear rate when `shear_type` is `linear`                                                        |
+-----------------+--------------------+------------------------------------------------------------------------------------------------------------+


``InitFlow`` section
--------------------

The ``InitFlow`` section defines the initial conditions of the flow. Several entries can be used simultaneously to define more complex initial conditions.

+------------------------+--------------------+-----------------------------------------------------------------------------------------------------------+
|  Entry name            | Parameter type     | Comment                                                                                                   |
+========================+====================+===========================================================================================================+
| large_scale_3d_noise   | float, float       | | (optional) Apply a 3D noise to all the wavemodes above a critical wavelength L.                         |
|                        |                    | | The first parameter is the noise amplitude, and the second parameter is the critical wavelength.        |
|                        |                    | | NB: the noise is applied to all fields.                                                                 |
+------------------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| large_scale_2d_noise   | float, float       | | (optional) Apply a 2D noise to all the wavemodes above a critical wavelength L in the (x,y) plane.      |
|                        |                    | | The first parameter is the noise amplitude, and the second parameter is the critical wavelength.        |
|                        |                    | | NB: the noise is applied to all fields.                                                                 |
+------------------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| large_scale_1d_noise   | float, float       | | (optional) Apply a 1D noise to all the wavemodes above a critical wavelength L in the (x) direction.    |
|                        |                    | | The first parameter is the noise amplitude, and the second parameter is the critical wavelength.        |
|                        |                    | | NB: the noise is applied to all fields.                                                                 |
+------------------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| shear_layer            | float, float       | | (optional) Create a shear layer:  :math:`v_x=-v_0` if :math:`y<y_0`, else :math:`v_x=v_0`               |
|                        |                    | | The first parameter is :math:`y_0` and the second parameter is :math:`v_0`.                             |
+------------------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| mean_flow              | float, float, float|  (optional) Adds a mean flow. The three parameters are the three components of the mean velocity          |
+------------------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| mean_field             | float, float, float| | (optional) Adds a constant magnetic field. The three parameters are the three components of field       |
|                        |                    | | :math:`\mathbf{B}=(B_x, B_y, B_z)`.                                                                     |
|                        |                    | | NB: this entry is only available when the rhs is ``mhd``.                                               |
+------------------------+--------------------+-----------------------------------------------------------------------------------------------------------+
| python                 | string, (string)   | | 1st parameter: Name of the Python function to call to initialize the flow in the script provided in the |
|                        |                    | | [python] block of the input file.                   ,                                                   |
|                        |                    | | 2nd parameter (optional): "real" or "fourier", to specify whether the field passed to the               |
|                        |                    | | Python function is in real space or Fourier space. Default is "real".                                   |
+------------------------+--------------------+-----------------------------------------------------------------------------------------------------------+

.. _astraPySection:

``Python`` section
--------------------
The python section defines how Astra interacts with the Python interpreter. Usually, Astra interacts
by calling dedicated python functions from a script provided in this section. The script is assumed
to be localised in the execution directory. The name of the function called inside the script are specified
in the dedicated items in the ``Output`` and ``InitFlow`` sections of the input file (see above).
The definition of the entries in the Python section is as follows:

.. note:: This functionality requires `Astra_PYTHON` to be enabled during the code configuration with Cmake.

+--------------------+-------------------------------------------------------------+
| Entry name         |  Comment                                                    |
+====================+=============================================================+
| ``script``         | filename of the python script, without the ".py" extension. |
+--------------------+-------------------------------------------------------------+

.. _outputSection:

``Output`` section
--------------------

+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
|  Entry name    | Parameter type          | Comment                                                                                          |
+================+=========================+==================================================================================================+
| dmp            | float                   | | 1st parameter: Code time interval between dump outputs, in code units.                         |
|                |                         | | If negative, dumps are disabled.                                                               |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
| dmp_dir        | string                  | | directory for dump file outputs. Default to "./"                                               |
|                |                         | | The directory is automatically created if it does not exist.                                   |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
| timevar        | string, string, ...     | | text file output of box-averaged variables. Each quantity is specified by a string:            |
|                |                         | | - ev: box-averaged kinetic energy  :math:`\langle \mathbf{v}^2/2 \rangle`                      |
|                |                         | | - vxmin/vymin/vzmin: minimum velocity component  :math:`\min_i |v_i|`                          |
|                |                         | | - vxmax/vymax/vzmax: maximum velocity component  :math:`\max_i |v_i|`                          |
|                |                         | | - w2: box-averaged enstrophy :math:`\langle (\nabla \times \mathbf{v})^2 /2\rangle`            |
|                |                         | | - vi.vj: box-averaged cross-correlation :math:`\langle v_i v_j \rangle` (i,j=x,y or z)         |
|                |                         | | - spectrum_vi.vj: shell-integrated power spectrum of the correlation :math:`v_i v_j`:          |
|                |                         | |   :math:`\int_{k-\Delta k/2}^{k+\Delta k/2} \langle \hat{v}_i(k) \hat{v}_j^*(k) \rangle dk`    |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
| timevar_dir    | string                  | | directory for timevar file outputs. Default to "./timevar"                                     |
|                |                         | | The directory is automatically created if it does not exist.                                   |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
| timevar_step   | float                   | | Time interval between timevar outputs, in code units.                                          |
|                |                         | | If negative, periodic timevar outputs are disabled.                                            |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
| vtk_dir        | string                  | | directory for vtk file outputs. Default to "./"                                                |
|                |                         | | The directory is automatically created if it does not exist.                                   |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
| vtk            | float                   | | Time interval between vtk outputs, in code units.                                              |
|                |                         | | If negative, periodic vtk outputs are disabled.                                                |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
| vtk_sliceN     | float, int, float,      | | Create VTK files that contain a slice (cut or average) of the full domain.                     |
|                | string                  | | the "N" of the entry name is an integer that identifies each slice, starting from n=1          |
|                |                         | | 1st parameter: Time interval between each slice VTK file                                       |
|                |                         | | 2nd parameter: plane of the slice. 0=(x2,x3) slice, 1=(x1,x3), 2=(x1,x2)                       |
|                |                         | | 3rd parameter: localisation of the slice (when the slice is an average, this parameter only    |
|                |                         | |                affects the localisation of the slice in the produced VTK file                  |
|                |                         | | 4th parameter: slice type. Can be "cut" (for a slice of the full domain) or "average" (for an  |
|                |                         | | average along the direction given by the second parameter). NB: "average" performs a naive     |
|                |                         | | point average, without any consideration of the cell volumes/areas.                            |
|                |                         | | NB2: this feature is in beta, and sometimes fails with some MPI implementations.               |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
| python         | float, string, (string) | | 1st parameeter: Time interval between python outputs, in code units.                           |
|                |                         | | 2nd parameter: Name of the Python function to call for output in the script provided in the    |
|                |                         | | [python] block of the input file.                   ,                                          |
|                |                         | | 3rd parameter (optional): "real" or "fourier", to specify whether the field passed to the      |
|                |                         | | Python function is in real space or Fourier space. Default is "real".                          |
+----------------+-------------------------+--------------------------------------------------------------------------------------------------+
