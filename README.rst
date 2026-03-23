SMRL
====

.. description.rst

**gym-coll** is a repository for building Markov decision processes which represent in colloidal assembly. Make *environments* by wrapping around *simuations* and feed them to *agents* to find control *policies*. It uses `hoomd-blue`_ to simulate colloidal particles with Monte Carlo or molecular dynamics, and the `gymnasium`_ framework to translate these, and other, simulations into the RL language.

.. _gymnasium: https://gymnasium.farama.org/ 
.. _hoomd-blue: https://hoomd-blue.readthedocs.io/en/latest/

.. intro.rst

Getting Started
===============

Prerequisites
*************

| gym-coll is easily set up using anaconda. Create a conda environment with the following packages installed.
| \- `numpy`_, `scipy`_: for numerical data anaysis
| \- `pytorch`_: to create neural network architectures
| \- `gsd`_: to store simulated particle trajectories

.. code-block:: bash

   $ conda create -n ENV_NAME python
   $ conda activate ENV_NAME
   $ conda install -c conda-forge numpy scipy gymnasium gsd

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


.. _numpy: https://numpy.org/doc/stable/
.. _scipy: https://docs.scipy.org/doc/scipy/
.. _pytorch: https://pytorch.org/docs/stable/
.. _pytorchrl: https://pytorchrl.readthedocs.io/en/latest/
.. _gymnasium: https://gymnasium.farama.org/ 
.. _hoomd-blue: https://hoomd-blue.readthedocs.io/en/latest/
.. _gsd: https://gsd.readthedocs.io/en/latest/
.. _installing hoomd-blue: https://hoomd-blue.readthedocs.io/en/latest/building.html
.. _freud: https://freud.readthedocs.io/en/latest/gettingstarted/installation.html
