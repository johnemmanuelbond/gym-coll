"""
Contains classes which inherit `gymnasium.Wrapper <https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper>`_ so that :doc:`environments <envs>` can interface with specific :doc:`simulations <sims>` properties, render their state to movies, or reset in a specific fasion.
"""
import numpy as np
import gc

import gymnasium as gym

from sims import Simbase
from sims.hpmc import Multipole as HPMC_Multipole
from sims.bd import Multipole as BD_Multipole

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


class OutOfBoxWrapper(gym.Wrapper):
    """For use with :py:mod:`envs.feedback_control`, this wrapper will end an episode early and return a large negative reward if the environment's underlying simulation allows a particle to leave the simulation box. This is unphysical in experiment.

    :param env: a gymnasium environment with an underlying simulation (usually hoomd) in the backend. This simulatiob must have the attribute :code:`in_box()` for this wrapper to detect if a particle has left the box.
    :type env: :py:class:`Env`
    :param box_reward: the reward to return if a particle has left the simulation box, defaults to -100
    :type box_reward: int, optional
    :raises AssertionError: if the underyling simulation can't return in/out of box information
    """        
    def __init__(self,env,box_reward = -20):
        assert hasattr(env.unwrapped, 'sim'), "underlying environment must run a simulation"
        assert isinstance(env.unwrapped.sim, Simbase), "underlying environment must run a simulation"
        assert hasattr(env.unwrapped.sim,'in_box') and callable(getattr(env.unwrapped.sim,'in_box')), "underlying simulation must have in_box method for this wrapper to do anything"
        self.env = env
        self._br = np.float32(box_reward)
        
    def step(self, action):
        """steps the environment forward under prescribed action. If a perticle has left the simulation box return a highly negative reward and truncate the episode.

        :param action: an action to pass backwards to the wrapped environment 
        :return: the environment's position in obsevration space, the reward for that position (with out-of-box considerations), whether the environment has terminated, whether the evironment has truncated, a dictionary of additional information
        :rtype: tuple[int|ndarray,float,bool,bool,dict]
        """        
        obs, reward, term, trunc, info = super().step(action)
        if not self.env.unwrapped.sim.in_box():
            reward = self._br
            trunc = True
            info = dict(info)               # changed here
            info["out_of_box_trunc"] = True # changed here
        else:
            info = dict(info)
            info["out_of_box_trunc"] = False # changed here

        return obs, reward, term, trunc, info



class BuckleWrapper(gym.Wrapper):
    """For use with :py:mod:`envs.feedback_control`, this wrapper will end an episode early and return a large negative reward if the environment's underlying simulation applies a field which would buckle particles out of plane. This is unphysical in experiment.

    :param env: a gymnasium environment with an underlying simulation (usually hoomd) in the backend. This simulation must have the attribute :code:`eta0` order for this wrapper to detect buckling.
    :type env: :py:class:`Env`
    :param k_buckle: the field strength in simulation units (kT) above which a highly concentrated ensemble of colloids will buckle
    :type k_buckle: scalar
    :param box_reward: the reward to return if a particle has left the simulation box, defaults to -100
    :type box_reward: int, optional
    :param eta_threshold: the area fraction above which a microstate is considered 'solid' enough to buckle, defaults to 0.8
    :type eta_threshold: scalar, optional
    :raises AssertionError: if the underyling simulation can't return buckling information (area fraction and threshold)
    """        
    def __init__(self,env,k_buckle, buckle_reward = -100, eta_threshold=0.8):
        assert hasattr(env.unwrapped, 'sim'), "underlying environment must run a simulation"
        assert isinstance(env.unwrapped.sim, HPMC_Multipole) or isinstance(env.unwrapped.sim, BD_Multipole), "underlying environment must run a Multipole simulation"
        assert hasattr(env.unwrapped.sim,'eta0'), "underlying simulation must have an eta0 method for this wrapper to do anything"
        self.env = env
        self._br = np.float32(buckle_reward)
        self._et = eta_threshold
        self._kb = k_buckle

    def step(self, action):
        """steps the environment forward under prescribed action. A field stronger than the crystallization voltage is applied while the simulation is condensed

        :param action: an action to pass backwards to the wrapped environment 
        :return: the environment's position in obsevration space, the reward for that position (with buckle considerations), whether the environment has terminated, whether the evironment has truncated, a dictionary of additional information
        :rtype: tuple[int|ndarray,float,bool,bool,dict]
        """        
        obs, reward, term, trunc, info = super().step(action)
        is_solid = self.env.unwrapped.sim.eta0 > self._et
        strong_field = np.any( self.env.unwrapped.sim.electrodes.k_trans > self._kb )
        if  is_solid and strong_field:
            reward = self._br
            trunc = True

        return obs, reward, term, trunc, info


class UpdateTimeWrapper(gym.Wrapper):
    """For use with :py:mod:`envs.feedback_control`, this wrapper will append the control update time to the action space of the underlying environment. This allows the agent to control how long to apply a given action for before updating again. The wrapper will also truncate the episode if the maximum elapsed time is exceeded.

    :param env: a gymnasium environment with an underlying simulation (usually hoomd) in the backend.
    :type env: :py:class:`Env`
    :param tspace: a gymnasium space object defining the time space to append to the action space of the underlying environment
    :type tspace: :py:class:`gymnasium.spaces.Space`
    :param time_set: if :code:`tspace` is a Discrete space, a list of times (in :math:`\\tau`, seconds, or MC sweeps) corresponding to each discrete index in :code:`tspace`
    :type time_set: list[scalar] | None, optional
    :param max_elapsed: the maximum elapsed time (in :math:`\\tau`, seconds, or MC sweeps) for an episode
    :type max_elapsed: scalar | None, optional
    :raises AssertionError: if the underyling simulation can't return runtime information
    :raises TypeError: if the time space is not a Discrete or Box space
    :raises NotImplementedError: if the action and time spaces are not both Discrete or both Box types
    """        
    def __init__(self,env,tspace:gym.spaces.Space, max_elapsed, time_set=None):
        assert hasattr(env.unwrapped, 'sim'), "underlying environment must run a simulation"
        assert isinstance(env.unwrapped.sim, HPMC_Multipole) or isinstance(env.unwrapped.sim, BD_Multipole), "underlying environment must run a Multipole simulation"
        assert hasattr(env.unwrapped.sim,'elapsed'), "underlying simulation must have an elapsed method for this wrapper to do anything"
        self.env = env

        disc_action = isinstance(env.action_space, gym.spaces.Discrete)
        cont_action = isinstance(env.action_space, gym.spaces.Box)
        if not (disc_action or cont_action): raise TypeError("underlying environment action space must be either Discrete or Box type")

        disc_time = isinstance(tspace, gym.spaces.Discrete)
        cont_time = isinstance(tspace, gym.spaces.Box)
        if cont_time: assert tspace.shape[0]==1, "if using a Box time space, it must be one-dimensional"
        if not (disc_time or cont_time): raise TypeError("time space must be either Discrete or Box type")

        if disc_time and disc_action:
            
            total_actions = tspace.n * env.action_space.n
            self.action_space = gym.spaces.Discrete(total_actions)
            idx = np.arange(total_actions)
            self.encoder = np.array([idx//env.action_space.n, idx%env.action_space.n]).T

        elif cont_time and cont_action:
            low = np.concatenate( (tspace.low, env.action_space.low) )
            high = np.concatenate( (tspace.high, env.action_space.high) )
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            raise NotImplementedError("action and time spaces must both be either Discrete or Box types")
        
        if disc_time:
            assert time_set is not None, "if using a discrete time space, user must specify a list of runtimes (in tau, s or MC sweeps) corresponding to each discrete index in tspace"
            self._tmap = lambda x: time_set[self.encoder[x][0]]
            self._amap = lambda x: self.encoder[x][1]
        elif cont_time:
            self._tmap = lambda x: x[0]
            self._amap = lambda x: x[1:]
        
        self._tmax = max_elapsed
        self._rt_init = env.unwrapped._rt

    def step(self, action):
        """steps the environment forward under prescribed action. The modified update time 

        :param action: an action to pass backwards to the wrapped environment 
        :return: the environment's position in obsevration space, the reward for that position (with buckle considerations), whether the environment has terminated, whether the evironment has truncated, a dictionary of additional information
        :rtype: tuple[int|ndarray,float,bool,bool,dict]
        """        
        dt = self._tmap(action)
        env_act = self._amap(action)
        dt_corrected = min(dt, self._tmax - self.env.unwrapped.sim.elapsed)
        self.env.unwrapped._rt = dt_corrected
        obs, reward, term, trunc, info = super().step(np.array(env_act).flatten())
        trunc = (self.env.unwrapped.sim.elapsed + dt) >= self._tmax

        scaled_reward = reward * dt_corrected/self._rt_init

        return obs, scaled_reward, term, trunc, info


# class BetterRender(gym.wrappers.RecordVideo):

#     def __init__(
#             self,
#             env,
#             video_folder: str,
#             episode_trigger = None,
#             step_trigger = None,
#             video_length: int = 0,
#             name_prefix: str = "rl-video",
#             fps: int | None = None,
#             disable_logger: bool = True,
#             sps=1):
        
#         super().__init__(env,
#                          video_folder=video_folder,
#                          episode_trigger=episode_trigger,
#                          step_trigger=step_trigger,
#                          video_length=video_length,
#                          name_prefix=name_prefix, #fps=fps,
#                          disable_logger=disable_logger)
#         self._s = int(sps)
    
#     def step(self, action):
#         for _ in range(self._s):
#             obs, reward, term, trunc, info = super().step(action)

#         # if (term or trunc) and self.recording:
#         #     super().stop_recording()

#         return obs, reward, term, trunc, info
    
#     def reset(self,seed=None,options=dict()):
#         gc.collect()
#         # if self.recording: super().stop_recording()
#         obs,info = super().reset(seed=seed,options=options)
#         return obs,info





# class ResetLDLD(gym.Wrapper):

#     def __init__(self, env, x0):
#         super().__init__(env)
#         self.env=env
#         self.x0 = x0
    
#     def reset(self,seed=None,options=dict()):
#         if options is None: options = dict()
#         options['x0'] = self.x0
#         obs,info =  self.env.reset(seed=seed, options=options)
#         return obs, info