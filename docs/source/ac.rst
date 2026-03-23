ac package
==========

Actor Critic is an RL algorithm which uses neural networks to approximate the policy, :math:`\pi`, and the value function, :math:`v_{\pi}`,  under that policy. Using *temporal difference*, actor critic algorithms update their approximate :math:`v_{\pi}` with respect to new *experience*, :math:`(s,a,r)`, generated using :math:`\pi`. Using *gradient descent* actor critic algorithms update their approximate, :math:`\pi`, in order to maximize :math:`v_{\pi}`.

ac.a2cd_torch module
--------------------

.. automodule:: ac.a2cd_torch
   :members:
   :undoc-members:
   :show-inheritance:

ac.a2cc_torch module
--------------------

.. automodule:: ac.a2cc_torch
   :members:
   :undoc-members:
   :show-inheritance:
