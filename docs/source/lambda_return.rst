lambda_return package
=====================

Create RL agent objects which inherit from :py:class:`LBase <lambda_return.lambase.LBase>` and can generate policies by updating the action-value Q-table using the :math:`\lambda`-return and temporal difference methods.

WIP

.. math::
    G(t:T) = \\Sum ...


base class
----------

.. autoclass:: lambda_return.lambase.LBase
   :members: assess, policy
   :undoc-members: train
   :show-inheritance:

lambda-return modules
---------------------

.. toctree::
   :maxdepth: 2

   lamsarsa
   lamq_learning