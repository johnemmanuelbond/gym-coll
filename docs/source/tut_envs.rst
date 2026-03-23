Environments
============

RL problems are framed in the context of a *Markov Decision Process* wherein an *agent* and an *environment* feed off of each other. In this tutorial we'll discuss what environments are, how to formulate colloidal physics in terms of an environment, and the functionality contained in SMRL for making modular environments for colloidal assembly problems.




Environments from Simulations
*****************************

Following the :doc:`Simulation <tut_sims>` tutorial, we can make a quick monte carlo simulation for 100 spherical colloids in a quadrupole. We'll characterize these particles using the :py:meth:`C6 <pchem.order.crystal_connectivity>` order parameter from the :doc:`pchem`.

.. code-block:: python

    import sims
    import pchem
    from utils import random_frame

    def C6(pts):
        #finds neighbors
        nei = pchem.order.neighbors(pts)
        #computes order parameter
        psis, _ = pchem.order.bond_order(pts,nei)
        C6s = pchem.order.crystal_connectivity(psis,nei)
        return (C6s.mean(),)

    L = 20
    monte = sims.hpmc.Quadrupole(100,C6,dg=L)
    init = random_frame(monte.num_particles,L)
    monte.reset(init_state=init) # again, it's a good idea to reset to an intial state before doing much else

Next we define the observation and action spaces, as well as the reward function. For this tutorial we'll bin C6 into 5 bins and choose one of 10 external potentials to apply to the ensemble. The environment takes in rewards and termination conditions as functions which act *on the observation space*. Since we're discretely binning a one dimensional C6 simulation, the observation space is \{0,1,2,3,4\}. The termination function should return a boolean if we've reached one of the termination states. In this case we'll call the last bin the terimation state.

.. code-block:: python

    import gymnasium as gym
    import numpy as np

    obs_space = gym.spaces.Discrete(5)
    act_space = gym.spaces.Discrete(10)
    # a set of quadratic coeffecients for the energy well that confines the particles, these correspond to experimental voltages
    actions = np.linspace(0,2.5,10)*L**2

    reward = lambda obs: 1*(obs==4) - 1*(obs!=4)
    terminate = lambda obs: obs in [4]

Finally, we can make and run the environment using gymnasium's make functionality, since we registerd this environment in the :code:`__init__.py` for the :doc:`envs`. Once the environment is made we can call :code:`step()` and :code:`reset()` as if we were the agent.

.. code-block:: python

    import envs # needed to register the environment via __init__.py

    env = gym.make('envs/fbc_discrete',
                sim=monte,
                observation_space=obs_space,
                action_space=act_space,
                lambda_reward=reward,
                lambda_terminate=terminate,
                action_set=actions,
                step_size=100) #we want each step to be 100 MC sweeps

    env.reset(options={"init_state": init})
    for _ in range(10):
        #step the environment at the lowest applied voltage
        o, r, t, _, _ = env.step(0)
    print(o, r, t)

    >>> 0, -1, False

Using the optional keyword argument :code:`options` in the :code:`reset()` method, we can pass arguments back to the simulation object. Here, we start an environment at a random state, condense it at a high voltage, and then reset to the end of that starting state and melt the system.

.. code-block:: python

    import gsd.hoomd
    #now if we specify an outfile, the simulaiton object will write particle trajectories
    env.reset(options={"init_state": init,'outfile':'hpmc_crystal.gsd'})
    for _ in range(20):
        # step the environment at the highest applied voltage
        env.step(9)

    #now when we reset we can specify the input file as the end of the crystal run and write to a new file to melt the crystal
    xtal = gsd.hoomd.open('hpmc_crystal.gsd',mode='r')[-1]
    env.reset(options={'init_state':xtal,'outfile':'hpmc_melting.gsd'})
    for _ in range(20):
        # step the environment at the lowest applied voltage
        env.step(0)

    # .gsd files are viewable in ovito, which is free to downlaod. So you can try this code yourself!

.. image:: media/hpmc_crystal.gif
   :width: 300
   :alt: hpmc_crystal.gsd: particles move towards center

.. image:: media/hpmc_melting.gif
   :width: 300
   :alt: hpmc_melting.gsd: particles move away from center

Mix and Match
*************

In the previous section we made one environment out of one simulation characterized by one order parameter characterization. However, simulations are in general characterized by a vector of order parameters. With this in mind, each environment in SMRL is designed to intelligently handle a simulation of arbitrary dimensionality. This makes the environments extremely modular, you can pair any physical simulation with any vector order parameter function with any kind of observation/action space.

.. code-block:: python

    #DEMO


Wrappers
********

.. code-block:: python

    #DEMO

Rendering
---------