"""
Contains classes which inherit `gymnasium.Wrapper <https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper>`_ so that :doc:`environments <envs>` can interface with specific :doc:`simulations <sims>` properties, render their state to movies, or reset in a specific fasion.
"""
import numpy as np

import gymnasium as gym

class GSDWrapper(gym.Wrapper):
    """
    A class used to make an environment interface with a :doc:`simulation <sims>` object such that :py:meth:`env.reset()` resets the underlying simulation to a user-defined initial state. Simultaneously, this class enables the environment to systematically save episodes to `gsd <https://gsd.readthedocs.io/en/stable/index.html>`_ files at certain episode triggers.

    :param env: a gymnasium environment, usually with a hoomd simulation in the backend
    :type env: :py:class:`Env`
    :param lib: a stack of gsd `Frames <https://gsd.readthedocs.io/en/stable/python-module-gsd.hoomd.html#gsd.hoomd.Frame>`_ to randomly choose from at each reset call
    :type lib: `HOOMDTrajectory <https://gsd.readthedocs.io/en/stable/python-module-gsd.hoomd.html#gsd.hoomd.HOOMDTrajectory>`_ | list[`Frame <https://gsd.readthedocs.io/en/stable/python-module-gsd.hoomd.html#gsd.hoomd.Frame>`_]
    :param trigger: a function which determines whether to save an episode to a `gsd`_ file at certain episode counts, defaults to Never
    :type trigger: functional, optional
    :param prefix: a prefix to add to the output file names, can be used to encode pathing within a directory structure, defaults to ''
    :type prefix: str, optional
    :param nsnap: the period (in seconds or MC sweeps) with which to record a simulation frame to the gsd file, defaults to 1.0
    :type nsnap: float, optional
    """        
    def __init__(self, env, gsd_lib, trigger = None, prefix='',nsnap=1.0):
        """
        Constructor
        """
        super().__init__(env)
        self.env=env
        self.gsd_lib = gsd_lib
        self._ep = 0
        self._snap = nsnap

        if trigger is None:
            self._trig = lambda i: False
            self._pf = None
        else:
            self._trig = trigger
            self._pf = prefix + '-'
    
    def reset(self,seed=None,options=dict()):
        """
        Resets the environment, and the underlying simulaiton, to an initial condition specifed by :code:`self.gsd_list`. If the trigger specified at instantiation returns true, the environment will write the episode to a `gsd`_ file.


        :param seed: RNG seed, defaults to None
        :type seed: int | None, optional
        :param options: kwargs for resetting the simulation, defaults to empty dict
        :type options: dict, optional
        :return: The position in observation space of the environment post reset, and a dictionary of additional information
        :rtype: tuple[float,dict]
        """        
        if options is None: options = dict()
        idx = np.random.randint(len(self.gsd_lib))
        options['init_state'] = self.gsd_lib[int(idx)]

        if self._trig(self._ep):
            if hasattr(self.env.unwrapped.sim, 'dt') and isinstance(self._snap,int):
                    nsnap = int(self._snap/self.env.unwrapped.sim.dt)
            else: nsnap = self._snap

            out = f"{self._pf}episode{self._ep:05}.gsd"
            options['outfile'] = out
            options['nsnap'] = nsnap
        else:
            options['outfile'] = None
            options['nsnap'] = None

        obs,info =  self.env.reset(seed=seed, options=options)
        self._ep+=1
        return obs, info
