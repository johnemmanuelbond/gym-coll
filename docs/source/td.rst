td package
==========

This package implements temporal difference (TD) learning algorithms for reinforcement learning. TD methods are a class of model-free algorithms that learn value functions by bootstrapping from current estimates, combining ideas from :doc:`Monte Carlo <rlmc>` methods and, more generally, dynamic programming.

**Core Concepts**

Temporal difference learning algorithms update value estimates based on the *temporal difference error* - the difference between the current estimate and a better estimate based on observed rewards and future values. All TD methods follow a general pattern:

.. math::

    \text{New Estimate} \leftarrow \text{Old Estimate} + \alpha[\text{Target} - \text{Old Estimate}]

where :math:`\alpha` is the learning rate and the *Target* varies depending on the specific algorithm.

For state-value methods, the update rule is:

.. math::

    V(s) \leftarrow V(s) + \alpha[R + \gamma V(s') - V(s)]

For action-value methods (Q-learning family), this generalizes to:

.. math::

    Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \cdot \text{Target}(s',a') - Q(s,a)]

where the *Target* component distinguishes different algorithms:

- **SARSA**: Uses :math:`Q(s',a')` where :math:`a'` is the action actually taken (on-policy)
- **Expected SARSA**: Uses :math:`\mathbb{E}[Q(s',a')]` over the policy distribution
- **Q-Learning**: Uses :math:`\max_{a'} Q(s',a')` regardless of policy (off-policy)  
- **Double Q-Learning**: Uses separate Q-tables to reduce maximization bias

**Hyperparameters**

- :math:`\alpha` (alpha): Learning rate controlling how much new information overrides old estimates
- :math:`\gamma` (gamma): Discount factor determining the importance of future rewards
- :math:`\epsilon` (epsilon): Exploration parameter for epsilon-greedy action selection

All algorithms in this package inherit from :py:class:`TDBase <td.tdbase.TDBase>` and implement these core TD principles with different target computation strategies.

base class
----------

.. autoclass:: td.tdbase.TDBase
   :members: assess_policy, get_best_action
   :undoc-members: get_policy
   :show-inheritance:

td modules
----------

.. toctree::
   :maxdepth: 2

   sarsa
   exp_sarsa
   q_learning
   doubleq_learning