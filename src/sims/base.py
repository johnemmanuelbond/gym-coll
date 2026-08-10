# -*- coding: utf-8 -*-
"""Base abstractions for colloidal simulations used by gym-coll.

This module defines the basic interface for simulation-backed environments.
A concrete simulation must be able to advance its state, reset itself to an
initial condition, and expose its current state as a vector of order
parameters.
"""
import string
from itertools import combinations_with_replacement as combinations
from warnings import warn
import importlib.util
import numpy as np
import gsd.hoomd

has_hoomd = False
try:
    spec = importlib.util.find_spec('hoomd')
    if spec is not None:
        has_hoomd=True
except ModuleNotFoundError:
    has_hoomd = False
    warn("hoomd not found, sims module may not work.")
if has_hoomd: import hoomd

from utils import SuperEllipse, random_frame

_default_sphere = SuperEllipse(ax=0.5,ay=0.5)
_alphabet = [f"A{i}" for i in range(100)]

class Simbase:
    """Minimal simulation interface for environments and wrappers.

    This base class is meant to be subclassed by concrete simulations. A
    subclass should define how its state is represented and how it advances and
    resets itself.
    """

    def __init__(self):
        """Initialize an empty simulation object."""
        pass


    @property
    def state(self) -> tuple:
        """
        :return: the position of the simulation in order parameter space
        :rtype: tuple
        """        
        return (None,)

    @property
    def elapsed(self)-> int:
        """
        :return: the total number of run calls since the last reset
        :rtype: int
        """
        return self.step

    @property
    def state_dim(self) -> int:
        """
        :return: the dimensionality of the simulation in order parameter space
        :rtype: int
        """        
        return len(self.state)

    def reset(self, **kwargs):
        """Reset the simulation to its initial configuration.

        :param kwargs: subclass-specific keyword arguments.
        :type kwargs: dict
        """
        self.step = 0

    def run(self, span, *args):
        """Advance the simulation for a short burst.

        :param span: length of the integration interval in simulation units
        :type span: scalar
        :param args: additional arguments passed to the simulation
        :type args: tuple
        """
        self.step += 1


class HoomdColloid(Simbase):
    """HOOMD-blue-backed colloidal simulation wrapper.

    This class manages a particle-based simulation with HOOMD-blue, including
    particle initialization, shape handling, and state computation from the
    current snapshot.

    :param dt: timestep for the Brownian integrator, defaults to 1e-3.
    :type dt: scalar, optional
    :param DT: short-time translational diffusivity in simulation units.
    :type DT: scalar, optional
    :param DR: short-time rotational diffusivity in simulation units.
    :type DR: scalar, optional
    :param kT: temperature in energy units.
    :type kT: scalar, optional
    :param num_types: number of particle types, defaults to 1.
    :type num_types: int, optional
    :param shape: particle shape, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>` object, optional
    """

    def __init__(self,
                 dt:float = 1e-3,
                 DT:float = 0.25,
                 DR:float = 0.025,
                 kT:float = 1.0,
                 num_types:int = 1,
                 shape:SuperEllipse = _default_sphere):
        """Initialize the simulation object and relevant global fields
        """
        super().__init__()

        self._N = 0
        self._sim_dims=2
        self._kT = kT
        self._DT = DT
        self._DR = DR
        self._dt = dt

        self._types = _alphabet[:num_types]

        # set shape and generate vertices if they're not alrady provided
        self.shape = shape
        self._logger = None

        seed = int(1000*np.random.rand())
        self._sim = hoomd.Simulation(device=hoomd.device.CPU(),seed=seed)

    def reseed(self, seed:int|None=None):
        """Reinitialize the simulation with a new random seed.

        :param seed: the new random seed to use for the simulation. defaults to a random integer if None is provided.
        :type seed: int | None
        """
        if seed is None: seed = int(1000*np.random.rand())
        self._sim = hoomd.Simulation(device=hoomd.device.CPU(),seed=seed)

    @property
    def frame(self) -> gsd.hoomd.Frame:
        """
        :return: the current simulation snapshot which contains particle position/orientation data
        :rtype: `gsd.hoomd.Frame <https://gsd.readthedocs.io/en/stable/python-module-gsd.hoomd.html#gsd.hoomd.Frame>`_
        """        
        if self._N == 0: return None
        return self._sim.state.get_snapshot()
    
    @property
    def dims(self) -> int:
        """
        :return: the dimensionality of the simulation in real space
        :rtype: int
        """        
        return self._sim_dims

    @property
    def num_particles(self) -> int:
        """
        :return: particle count
        :rtype: int
        """        
        return self._N

    @property
    def num_types(self) -> int:
        """
        :return: the number of particle types in the simulation
        :rtype: int
        """        
        return len(self._types)
    
    @num_types.setter
    def num_types(self, num_types:int):
        """
        :param num_types: the number of particle types in the simulation
        :type num_types: int
        """        
        self._types = _alphabet[:num_types]

    @property
    def types(self) -> list:
        """
        :return: the list of particle types in the simulation
        :rtype: list
        """        
        return self._types

    @types.setter
    def types(self, types:list):
        """
        :param types: the list of particle types in the simulation
        :type types: list
        """        
        self._types = types

    @property
    def pairs(self) -> list:
        """
        :return: the list of all unique pairs of particle types in the simulation
        :rtype: list
        """
        return [p for p in combinations(self.types,2)]
    
    @property
    def box(self) -> list:
        """
        :return: simulation box
        :rtype: array-like
        """        
        return self.frame.configuration.box
    
    @property
    def shape(self) -> SuperEllipse:
        """
        :return: the current shape used in the simulation
        :rtype: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>`
        """        
        return self._s
    
    @shape.setter
    def shape(self,shape:SuperEllipse):
        """
        :param shape: the shape to be used in the simulation
        :type shape: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>`
        """        
        self._s = shape
        self._is_disc = np.round(shape.aspect,4)==1.0 and np.round(shape.n,4)==2.0
        if (not self._is_disc) and (not hasattr(shape,'vertices')):
            self._s.contact_vertices(n_verts=12, require_corners=True)

    @property
    def elapsed(self) -> float:
        """
        :return: the total elasped time (in simulation units) since the last reset
        :rtype: scalar
        """
        return self._time

    @property
    def kT(self) -> float:
        """
        :return: the temperature in simulation energy units
        :rtype: scalar
        """        
        return self._kT

    @kT.setter
    def kT(self,kT:float):
        """
        :param kT: the temperature in simulation energy units
        :type kT: scalar
        """        
        self._kT = kT

    @property
    def DT(self) -> float:
        """
        :return: particles' short-time translational diffusivity in simulation units
        :rtype: scalar
        """        
        return self._DT
    
    @DT.setter
    def DT(self,DT:float):
        """
        :param DT: particles' short-time translational diffusivity in simulation units
        :type DT: scalar
        """        
        self._DT = DT

    @property
    def DR(self) -> float:
        """
        :return: particles' short-time rotational diffusivity in simulation units
        :rtype: scalar
        """        
        return self._DR
    
    @DR.setter
    def DR(self,DR:float):
        """
        :param DR: particles' short-time rotational diffusivity in simulation units
        :type DR: scalar
        """        
        self._DR = DR
    
    @property
    def dt(self)-> float:
        """
        :return: integration timestep in simulation units
        :rtype: scalar
        """        
        return self._dt

    @property
    def state(self) -> tuple:
        """Return the current state vector derived from the active snapshot. Must be overrriden in subclasses to actually use in environments.

        :return: the vector of order parameters computed by the state functional
        :rtype: tuple
        """
        warn("state property must be implemented in subclasses")
        return (None,)


    # @property
    # def state(self) -> tuple:
    #     """Return the current state vector derived from the active snapshot.

    #     :return: the vector of order parameters computed by the state functional
    #     :rtype: tuple
    #     """
    #     frame = self.frame
    #     pts = frame.particles.position
    #     if self._is_disc:
    #         os = None
    #     else:
    #         os = frame.particles.orientation
    #     state = self._lambda_f(pts,os,self._s)
    #     if isinstance(state,tuple): return state
    #     if isinstance(state,list): return tuple(state)
    #     if isinstance(state, int) or isinstance(state, float) or isinstance(state,np.float): return (state,)
    #     if isinstance(state, np.ndarray): return tuple(state.tolist())
    #     return state

    @property
    def logger(self) -> hoomd.logging.Logger | None:
        """Return a HOOMD logger for recording simulation quantities.

        :return: a HOOMD logger object that records information about the simulation run
        :rtype: `hoomd.logging.Logger <https://hoomd-blue.readthedocs.io/en/stable/hoomd/logging/logger.html>`_
        """
        if self._logger is None:
            self._logger = hoomd.logging.Logger(only_default=False)
        return self._logger

    @logger.setter
    def logger(self, logger:hoomd.logging.Logger):
        """Set the HOOMD logger for recording simulation quantities.

        :param logger: a HOOMD logger object that records information about the simulation run
        :type logger: `hoomd.logging.Logger <https://hoomd-blue.readthedocs.io/en/stable/hoomd/logging/logger.html>`_
        """
        self._logger = logger

    @property
    def sim(self) -> hoomd.Simulation:
        """Return the current HOOMD simulation object.

        :return: the HOOMD simulation object that manages the particle dynamics
        :rtype: `hoomd.Simulation <https://hoomd-blue.readthedocs.io/en/stable/hoomd/simulation/simulation.html>`_
        """
        return self._sim

    @property
    def integrator(self):
        """Return the current HOOMD integrator object.
        """
        raise NotImplementedError("integrator property must be implemented in subclasses")

    def add_updater(self, updater:hoomd.update.Update):
        """Add a HOOMD updater to the simulation.

        :param updater: a HOOMD updater object that modifies the simulation state
        :type updater: `hoomd.update.Update <https://hoomd-blue.readthedocs.io/en/stable/hoomd/update/update.html>`_
        """
        self._sim.operations.updaters.append(updater)

    def reset(self,
              N_random:int|None = None,
              init_state:gsd.hoomd.Frame | None = None,
              outfile:str | None = None,
              nsnap:float = 0.1,
              seed: int | None = None,
              mode:str = 'ab',
              ):
        """Reset the simulation and rebuild the HOOMD state.

        This creates a fresh HOOMD simulation, reinitializes the particles from
        either a provided frame or a generated random state, and optionally
        writes snapshots to an output GSD file.

        :param N_random: the number of particles to randomly initialize if no initial state is provided. Defaults to None, which means that the number of particles will be determined by the provided initial state.
        :type N_random: int | None, optional
        :param init_state: particle configuration to use as the initial state. Defaults to a random state created with :py:func:`utils.hoomd_helpers.random_frame`.
        :type init_state: gsd.hoomd.Frame or None, optional
        :param outfile: path to an output GSD file for recording particle configurations. If omitted, no trajectory is written.
        :type outfile: str or None, optional
        :param nsnap: snapshot period in simulation units used when writing the output trajectory.
        :type nsnap: scalar, optional
        :param seed: random seed used to reinitialize the simulation.
        :type seed: int or None, optional
        :param mode: file mode for the output GSD file. Use 'ab' to append, 'wb' to overwrite, or 'xb' to create a new file and fail if it already exists.
        :type mode: str, optional
        """
        super().reset()
        assert has_hoomd, "hoomd-blue not found, install hoomd-blue to use this module."
        self._time = 0

        #remove all writers to close out current gsd file so that subsequent steps continue to append frames.
        for op in self._sim.operations:
            self._sim.operations.remove(op)

        #load initial state from frame object into new simulation object
        self.reseed(seed)

        #load initial state from frame object into new simulation object
        if init_state is None:
            if N_random is not None: self._N = N_random
            init = random_frame(self._N,2*int(np.sqrt(self._N)*self._s.ax),shape=self._s, types=self._types)
        else:
            self._N = init_state.particles.N
            init = gsd.hoomd.Frame()
            init.configuration.box = init_state.configuration.box
            init.particles.N = self._N

            n_init_types = len(np.unique(init_state.particles.typeid))
            self.num_types = max(self.num_types,n_init_types)
            init.particles.types = self.types

            init.particles.position = init_state.particles.position
            init.particles.orientation = init_state.particles.orientation
            init.particles.typeid = init_state.particles.typeid
            init.particles.image = np.zeros_like(init.particles.position)
            init.particles.moment_inertia = init_state.particles.moment_inertia

        self._sim_dims = 3 - (init.configuration.box[2]==0)
        self._sim.create_state_from_snapshot(init)
        
        #define file writer which continually appends simulation bursts to a trajectory file
        if not (outfile is None):
            gsd_writer = hoomd.write.GSD(filename=outfile,
                                    trigger=hoomd.trigger.Periodic(int(nsnap/self._dt)),
                                    mode=mode,
                                    dynamic=['property','momentum','attribute'])
            self._sim.operations.writers.append(gsd_writer)

    def run(self,
            time:float,
            *args):
        """Advance the simulation for a short burst.

        :param time: runtime in simulation units for this burst
        :type time: scalar
        :param args: additional arguments passed to the simulation
        :type args: tuple
        """
        super().run(time, *args)

        self._time+=time

        # add field strength to logger so that this quantity is associated with each frame
        if len(self._sim.operations.writers)>0:
            gsd_writer = self._sim.operations.writers[0]
            gsd_writer.logger = self.logger
            
        # apply forces to integrator and run for one simstep
        self._sim.operations.integrator = self.integrator
        simstep = int(time/self._dt)
        self._sim.run(simstep)
        if len(self._sim.operations.writers)>0:
            self._sim.operations.writers[0].flush()
        self._sim.operations.integrator = None


if __name__ == "__main__":
    pass