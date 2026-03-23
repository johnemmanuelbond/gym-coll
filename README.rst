SMRL
====

.. description.rst

**S**\ oft **M**\ atter **R**\ einforcement **L**\ earning is a repository built to use reinforcement learning on control problems in colloidal assembly. Make *environments* by wrapping around *simuations* and feed them to *agents* to find control *policies*. It uses `hoomd-blue`_ to simulate colloidal particles with Monte Carlo or molecular dynamics, and the `gymnasium`_ framework to translate these, and other, simulations into the RL language.

.. _gymnasium: https://gymnasium.farama.org/ 
.. _hoomd-blue: https://hoomd-blue.readthedocs.io/en/latest/

.. intro.rst

Getting Started
===============

Prerequisites
*************

| SMRL is easily set up using anaconda. Create a conda environment with the following packages installed.
| \- `numpy`_, `scipy`_: for numerical data anaysis
| \- `gymnasium`_, `pytorchrl`_: to inherit modular features for RL environments. gym version 1.0.0+ is highly recommended, it may not be the default, and sometimes v.0.28.1 is necessary to avoid errors
| \- `pytorch`_: to create neural network architectures
| \- `gsd`_: to store simulated particle trajectories

.. code-block:: bash

   $ conda create -n ENV_NAME python=3.12
   $ conda activate ENV_NAME
   $ conda install -c conda-forge numpy scipy gymnasium=1.0 pytorch pytorchrl gsd

Additionally, `installing hoomd-blue`_ is required. Once compiled, users may add hoomd to their SMRL conda environment regardless of compilation details specicially. Alternatively, users may use a pre-compiled version of hoomd elsewhere on their system.

.. code-block:: bash

   $ conda develop path_to_hoomd/build

Many :doc:`simulations <sims>` are built for homebrewed versions of hoomd where the source code has been edited to achieve desired physics. The edited files are included in the github repository, but users unfamiliar with hoomd should use a precompiled version from a colleague.

Installation
************

After installing prerequesites, clone the git repository:

.. code-block:: bash

   $ git clone https://github.com/dayakaran/SMRL.git path_to_repo

Users may install the repository into a conda environment using :code:`conda develop`

.. code-block:: bash

   $ conda develop path_to_repo/src

Uninstall with

.. code-block:: bash

   $ conda develop -u path_to_repo/src

Additional Notes
****************

Bevan group users can call :code:`conda develop /group/Bond/code/hoomd-v5-npole/build` to install a homebrewed version of version 5.2 where :class:`hoomd.md.force.Active` and :class:`hoomd.hpmc.external.Harmonic` have been modified to model an multipolar electrode for use with the :py:mod:`hmpc <sims.hpmc>` module. *requires python version 3.12*

Hoomd may be conifugured to run with mpi. Installing openmpi into your conda environment is necessary to use this feautre in SMRL.

| Addtionally, `freud`_ is a useful module for characterizing colloidal ensembles if users find that the tools provided are insufficient for their purposes.
| For visualization install matplotlib and ffmpeg (at least).
| For editing and compiling documentation, install sphinx and sphinx-rtd-theme


.. _numpy: https://numpy.org/doc/stable/
.. _scipy: https://docs.scipy.org/doc/scipy/
.. _pytorch: https://pytorch.org/docs/stable/
.. _pytorchrl: https://pytorchrl.readthedocs.io/en/latest/
.. _gymnasium: https://gymnasium.farama.org/ 
.. _hoomd-blue: https://hoomd-blue.readthedocs.io/en/latest/
.. _gsd: https://gsd.readthedocs.io/en/latest/
.. _installing hoomd-blue: https://hoomd-blue.readthedocs.io/en/latest/building.html
.. _freud: https://freud.readthedocs.io/en/latest/gettingstarted/installation.html
