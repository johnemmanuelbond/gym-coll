# -*- coding: utf-8 -*-
"""
Contains environment subclasses that broadly pertain to feedback control problems.

Environments inherit the gymnasium base class and so must contain a step() method, which returns a state and reward based on an action, and a reset() method, which resets the environment back to some initial configuration.

Additionally, environments have a predefined observation space and action space so that agents can clearly define value and action-value functions over 

Environments calculate a reward at the end of each step. In the SMRL framework the reward is a functional which acts over the observation space at the end of each step.

In episodic RL problems environments can temrinate if they meet a condition. Similarly to the reward, the termination condition is a functional which acts over observation space.

Each of these subclasses accepts in it's constructor a :py:mod:`simulation <sims.base>` object. Simulations have a sim.state property which returns a vector of order prameters. Simulations also have a run(span, action) function which accepts a span of time/sweeps to run and an action such as a voltage. Finally, simulations have a reset() method which reinstantiates an initial condition.

"""

import numpy as np
import gymnasium as gym

from gymnasium import spaces

from sims import base

class Discrete(gym.Env):
    """An environment with a discrete observation space and a discrete action space. The dimensionality of the observation space and the underlying simulation must match. Additionally, provide a mapping for action which correspond to discrete indices in the action space.

    :param sim: An arbitrary simulation object
    :type sim: base.Simbase
    :param observation_space: The discrete observation space with dimensions that match the dimensionality of the simulation
    :type observation_space: gymnasium.spaces.Discrete | spaces.MultiDiscrete
    :param action_space: The discrete action space
    :type action_space: gymnasium.spaces.Discrete
    :param lambda_reward: The reward functional which acts over observation space
    :type lambda_reward: function
    :param lambda_terminate: The termination function which acts over observation space
    :type lambda_terminate: function
    :param action_set: The map from values in the action space to the physical actions needed for the simulation
    :type action_set: array_like
    :param max_steps: the number of steps allowed before tuncation, defines a maximum episode length, defaults to 100
    :type max_steps: int, optional
    :param step_size: the length (in time or sweeps) of the simulaiton burst in :py:meth:`step`, defaults to 1
    :type step_size: scalar, optional
    """        

    def __init__(self,
                 sim:base.Simbase,
                 observation_space: spaces.Discrete | spaces.MultiDiscrete,
                 action_space: spaces.Discrete,
                 lambda_reward,
                 lambda_terminate,
                 action_set,
                 max_steps:int=100,
                 step_size:float|int=1,
                 ):
        """
        Constructor
        """
        self.sim = sim

        self.action_space = action_space
        self.observation_space = observation_space

        if isinstance(observation_space, spaces.Discrete):
            assert sim.dims == 1, "Simulation dimensions must match observation space"
        else:
            assert len(observation_space.shape) == 1, "observation space must have at most rank 1"
            assert sim.dims == observation_space.shape[0], "Simulation dimensions must match observation space"
        
        assert action_space.n == len(action_set), "Set of actions must have same shape as action space"
        

        self._R = lambda_reward
        self._T = lambda_terminate
        self._end = max_steps
        self._act = action_set
        self._rt = step_size

    @property
    def actions(self) -> list|np.ndarray:
        """
        :return: the list of actions which correspond to the indices of the discrete action space
        :rtype: ndarray or list
        """
        return self._act

    def _get_obs(self) -> int|np.ndarray:
        """
        :return: The current position in observation space
        :rtype: int | ndarray
        """        

        if self.sim.dims==1:
            n = self.observation_space.n
            op = self.sim.state[0]
            obs = min(int(op * n),n-1)
        else:
            nvec = self.observation_space.nvec
            ovec = self.sim.state
            obs = tuple([min(int(op * n),n-1) for op,n in zip(ovec,nvec)])
            # obs = np.array(tuple([min(int(op * n),n-1) for op,n in zip(ovec,nvec)]))
        
        return obs
    
    def _get_info(self) -> dict:
        """
        :return: extra info about the state. By default includes elapsed steps and elapsed time if available.
        :rtype: dict
        """        
        info = dict()
        info['elapsed_steps'] = self.sim.step
        info['elapsed_time'] = self.sim.elapsed
        return info

    def is_terminal(self, obs:int|np.ndarray) -> bool:
        """checks if an observation is a terminal state

        :param obs: a point in the observation space
        :type obs: int | ndarray
        :return: Whether the observation is terminal
        :rtype: bool
        """        
        return self._T(obs)

    def reset(self,
              seed: int | None = None,
              options:dict = dict()) -> tuple[float,dict]:
        """Resets the environment, and the underlying simulaiton, to an initial condition.

        :param seed: RNG seed, defaults to None
        :type seed: int | None, optional
        :param options: kwargs for resetting the simulation, defaults to empty dict
        :type options: dict, optional
        :return: The position in observation space of the environment post reset, and a dictionary of additional information
        :rtype: tuple[float,dict]
        """        
        if options is None: options = dict()

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.sim.reset(seed=seed,**options)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self,action:int|tuple|np.ndarray) -> tuple[int|np.ndarray,float,bool,bool,dict]:
        """Steps the environment forward by running the underlying simulation for a short burst and recording the resulting order parameters.

        :param action: the index of the desired action in the action space
        :type action: int | tuple | ndarray
        :return: the environment's position in obsevration space, the reward for that position, whether the environment has terminated, whether the evironment has truncated, a dictionary of additional information
        :rtype: tuple[int|ndarray,float,bool,bool,dict]
        """        
        self.sim.run(self._rt, *np.array([self._act[action]]).flatten())

        obs = self._get_obs()
        info = self._get_info()
        reward = self._R(obs)
        term = self._T(obs)
        trunc = self.sim.step > self._end
        
        return obs, reward, term, trunc, info





class Semidiscrete(gym.Env):
    """An environment with a continuous observation space and a discrete action space. The dimensionality of the observation space and the underlying simulation must match. Additionally, provide a mapping for action which correspond to discrete indices in the action space.

    :param sim: An arbitrary simulation object
    :type sim: base.Simbase
    :param observation_space: The continuous observation space whose dimensionality must match the dimensionality of the simulation
    :type observation_space: gymnasium.spaces.Box
    :param action_space: The discrete action space
    :type action_space: gymnasium.spaces.Discrete
    :param lambda_reward: The reward functional which acts over observation space
    :type lambda_reward: function
    :param lambda_terminate: The termination function which acts over observation space
    :type lambda_terminate: function
    :param action_set: The map from values in the action space to the physical actions needed for the simulation
    :type action_set: array_like
    :param max_steps: the number of steps allowed before tuncation, defines a maximum episode length, defaults to 100
    :type max_steps: int, optional
    :param step_size: the length (in time or sweeps) of the simulaiton burst in :py:meth:`step`, defaults to 1
    :type step_size: scalar, optional
    """        

    def __init__(self,
                 sim:base.Simbase,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 lambda_reward,
                 lambda_terminate,
                 action_set,
                 max_steps:int=100,
                 step_size:float|int=1,
                 ):
        """
        Constructor
        """
        self.sim = sim

        self.action_space = action_space
        self.observation_space = observation_space
        
        assert len(observation_space.shape) == 1, "observation space must have at most rank 1"
        assert sim.dims == observation_space.shape[0], "Simulation dimensions must match observation space"
        assert action_space.n == len(action_set), "Set of actions must have same shape as action space"

        self._R = lambda_reward
        self._T = lambda_terminate
        self._end = max_steps
        self._act = action_set
        self._rt = step_size
    
    @property
    def actions(self) -> list|np.ndarray:
        """
        :return: the list of actions which correspond to the indices of the discrete action space
        :rtype: ndarray or list
        """
        return self._act

    def _get_obs(self) -> int|tuple:
        """
        :return: The current position in observation space
        :rtype: int|tuple
        """        
        return np.array(self.sim.state,dtype=np.float32)
    
    def _get_info(self) -> dict:
        """
        :return: extra info about the state. By default includes elapsed steps and elapsed time if available.
        :rtype: dict
        """        
        info = dict()
        info['elapsed_steps'] = self.sim.step
        info['elapsed_time'] = self.sim.elapsed
        return info
    
    def is_terminal(self, obs:int|tuple) -> bool:
        """checks if an observation is a terminal state

        :param obs: a point in the observation space
        :type obs: int | tuple
        :return: Whether the observation is terminal
        :rtype: bool
        """        
        return self._T(obs)

    def reset(self,
              seed: int | None = None,
              options:dict = dict()) -> tuple[float,dict]:
        """Resets the environment, and the underlying simulaiton, to an initial condition.

        :param seed: RNG seed, defaults to None
        :type seed: int | None, optional
        :param options: kwargs for resetting the simulation, defaults to empty dict
        :type options: dict, optional
        :return: The position in observation space of the environment post reset, and a dictionary of additional information
        :rtype: tuple[float,dict]
        """        

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.sim.reset(seed=seed,**options)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self,action:int|tuple|np.ndarray) -> tuple[int,float,bool,bool,dict]:
        """Steps the environment forward by running the underlying simulation for a short burst and recording the resulting order parameters.

        :param action: the index of the desired action in the action space
        :type action: int | tuple | np.ndarray
        :return: the environment's position in obsevration space, the reward for that position, whether the environment has terminated, whether the evironment has truncated, and a dictionary of additional information
        :rtype: tuple[int,float,bool,bool,dict]
        """        
        self.sim.run(self._rt, *np.array([self._act[action]]).flatten())

        obs = self._get_obs()
        info = self._get_info()
        reward = self._R(obs)
        term = self._T(obs)
        trunc = self.sim.step > self._end
        
        return obs, reward, term, trunc, info





class Continuous(gym.Env):
    """An environment with a continuous observation space and a discrete action space. The dimensionality of the observation space and the underlying simulation must match. The action space should span the space of physical actions (no indexing).

    :param sim: An arbitrary simulation object
    :type sim: base.Simbase
    :param observation_space: The continuous observation space whose dimensionality must match the dimensionality of the simulation.
    :type observation_space: gymnasium.spaces.Box
    :param action_space: The continuous action space
    :type action_space: gymnasium.spaces.Box
    :param lambda_reward: The reward functional which acts over observation space
    :type lambda_reward: function
    :param lambda_terminate: The termination function which acts over observation space
    :type lambda_terminate: function
    :param max_steps: the number of steps allowed before tuncation, defines a maximum episode length, defaults to 100
    :type max_steps: int, optional
    :param step_size: the length (in time or sweeps) of the simulaiton burst in :py:meth:`step`, defaults to 1
    :type step_size: scalar, optional
    """        

    def __init__(self,
                 sim:base.Simbase,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 lambda_reward,
                 lambda_terminate,
                 max_steps:int=100,
                 step_size:float|int = 1,
                 ):
        """
        Constructor
        """
        self.sim = sim

        self.action_space = action_space
        self.observation_space = observation_space

        assert len(observation_space.shape) == 1, "observation space must have at most rank 1"
        assert sim.dims == observation_space.shape[0], "Simulation dimensions must match observation space"

        self._R = lambda_reward
        self._T = lambda_terminate
        self._end = max_steps
        self._rt = step_size
        
        self.field_low  = self.action_space.low
        self.field_high = self.action_space.high
        
        
    def _get_obs(self) -> int|tuple:
        """
        :return: The current position in observation space
        :rtype: int|tuple
        """        
        return np.array(self.sim.state,dtype=np.float32)

    def _get_info(self) -> dict:
        """
        :return: Extra information about the state. By default includes elapsed steps and elapsed time if available.
        :rtype: dict
        """        
        info = dict()
        info['elapsed_steps'] = self.sim.step
        info['elapsed_time'] = self.sim.elapsed
        return info
    
    def is_terminal(self, obs:int|tuple) -> bool:
        """checks if an observation is a terminal state

        :param obs: a point in the observation space
        :type obs: int | tuple
        :return: Whether the observation is terminal
        :rtype: bool
        """        
        return self._T(obs)

    def reset(self,
              seed: int | None = None,
              options:dict = dict()) -> tuple[float,dict]:
        """Resets the environment, and the underlying simulaiton, to an initial condition.

        :param seed: RNG seed, defaults to None
        :type seed: int | None, optional
        :param options: kwargs for resetting the simulation, defaults to empty dict
        :type options: dict, optional
        :return: The position in observation space of the environment post reset, and a dictionary of additional information
        :rtype: tuple[float,dict]
        """        

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.sim.reset(seed=seed,**options)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self,action:float|tuple|np.ndarray) -> tuple[int,float,bool,bool,dict]:
        """Steps the environment forward by running the underlying simulation for a short burst and recording the resulting order parameters.

        :param action: the desired action
        :type action: scalar, array-like
        :return: the environment's position in obsevration space, the reward for that position, whether the environment has terminated, whether the evironment has truncated, and a dictionary of additional information
        :rtype: tuple[int,float,bool,bool,dict]
        """        
        self.sim.run(self._rt, *np.array([action]).flatten())

        obs = self._get_obs()
        info = self._get_info()
        reward = self._R(obs)
        term = self._T(obs)
        trunc = self.sim.step > self._end
        
        return obs, reward, term, trunc, info
