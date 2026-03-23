# -*- coding: utf-8 -*-
"""
A group of simuation classes meant for simulating quasi-2D ensembles of hard colloidal particles within various electrode geometries. The main :py:class:`Multipole` class can hanle any particle geometry and any electrode geometry, but there are also subclasses designed to simulate specific experimental cases, spheres in a quadrupole, spheres in an octopole, and superellipses (ellipses/rectangles) between coplanar electrodes. Each of these classes is written to function in common 'simulation units' which is a nondimensionalization scheme to report energies, and lengths without referring to experimental conditions.

- Energy units: the thermal energy scale :math:`kT`
- Time units :math:`\\tau`: the time it takes for a single colloidal particle to *diffuse* a length unit (on average). Technically this unit gets set by the choice of :math:`dt` in the Brownian Dynamics equation of motion. Practically, the user defines this value through the choice of a free-particle diffusion constant, :math:`D_0`. The default value of :math:`D_0 = 1 [l^2/\\tau]` means the *mean-squared displacement* curve for a noninteracting particle will have a slope of :math:`4 [l^2/\\tau]` in two dimensions.\n
- Length units: the diameter of spherical colloids :math:`l=2a`, or the short axis of anisotropic colloids :math:`l=2a_y`.

Brownian dynamics is able to simulate "soft" colloidal particles which interact through chemical physics like electrostatic double layers. So while these classes use the :py:class:`SuperEllipse <utils.geometry.SuperEllipse>` class to represent numerous particle geometries including spheres, ellipses, rectangles, and rhombuses, they also involve force-fields from the `hoomd.md.pair <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/module-pair.html>`_ module.

Particles in a multipolar electode experience a harmonic confining force which drives particles towards the center of the electode: :math:`U=\\frac{{1}}{{2}}kx^2`. We characterize this harmonic confinement with a 'spring constant', 'quadratic coeffecient', 'harmonic trap strength', 'field-strength', etc given by :math:`k`, with units :math:`[kT]`. Negative field strengths correspond to driving the particles towards the electrode edges. This module makes heavy use of the :py:class:`Electrodes <utils.hoomd_helpers.Electrodes>` class to translate field strengths along sets of arbitrary directions into actionable hoomd calls. To properly use this module, ensure that the linked version of hoomd has the `ExternalFieldHarmonic.h` C++ header modifed as shown in the hoomd-mods/hoomd-v5-npole subdirectory of the SMRL repository.

It is simple to use the :py:mod:`pchem.units <pchem.units>` module with a set a set of experimental conditions to convert from these simulation units back to experimentally accessible micrometers, kelvin, and volts. Namely, the field strength is a simple monotonic function of the voltage. So converting an applied voltage, and so calculating the correct field strength to use in simulation units is a simple as a function call from :py:mod:`pchem.units <pchem.units>`.
"""

import numpy as np
import gsd.hoomd

import importlib.util
has_hoomd = False
try:
    spec = importlib.util.find_spec('hoomd')
    if spec is not None:
        has_hoomd=True
except ModuleNotFoundError:
    has_hoomd = False
    raise Warning("hoomd not found, sims.bd module will not work. Install hoomd-blue to use this module.")
if not has_hoomd: import hoomd

from sims import Simbase
from utils import SuperEllipse, Electrodes, random_frame
from utils import hoomd_wca, hoomd_alj

_default_sphere = SuperEllipse(ax=0.5,ay=0.5)
_default_qpole = Electrodes(n=2,dg=30)
_default_qpole.direct = np.array([0,np.pi/2])

class Multipole(Simbase):
    """
    A generic class for 2D Brownian Dynamics simulations of shaped hard particles in any multipolar electrode configuration.

    Multipole simulations are for generic shapes. Therefore the :code:`state_functional` parameter of the constructor must be a python function which takes as arguments the particle positions (Nx3 array), the quaternion orientations of the particles (Nx4 array) (see :py:meth:`quat_to_angle <utils.geometry.quat_to_angle>` to convert), and the simulation's :code:`shape` property (:py:class:`SuperEllipse <utils.geometry.SuperEllipse>`).

    Multipole simulations contain an attribute :py:attr:`Multipole.interaction` which allows access to the underlying `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/pair.html>`_ object. This attribute is settable so that users can employ functions from the :py:mod:`util.hoomd_helpers` module to to determine particle-particle interactions, or code specific use cases themselves. For anisotropic particles specifically, we approximate hard-particle behavior using a `hoomd.md.pair.aniso.ALJ <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/aniso/alj.html>`_ class.

    Multipole simulations also contain an attribute :py:attr:`Multipole.methods` which allows access to the underlying `hoomd.md.methods.Method <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/methods/method.html>`_ object. This attribute is settable so that users can employ any integration method they choose (including RATTLE for curved surfaces), but defaults to Brownian Dynamics with the simulation's current settings for translational and rotational diffusivity.

    :param N: the number of particles in the simulation, which may be modified with :py:meth:`reset()`
    :type N: int
    :param state_functional: A function of which reads the simulation microstate (positions, orientations) and returns the desired vector of order parameters as a tuple.
    :param shape: a particle shape, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>` object, optional
    :param electrodes: en electrode configuration, defaults to a symmetric quadrupole
    :type electrodes: :py:class:`Electrodes <utils.hoomd_helpers.Electrodes>`, optional
    :param dt: timestep for the brownian integrator, defaults to 2e-4
    :type dt: scalar, optional
    :param DT: Particles' short-time translational diffusivity (in simulation units), defaults to 0.25 which produces a 1-1 MSD in 2D
    :type DT: scalar, optional
    :param DR: Particles' short-time rotational diffusivity (in simulation units), defaults to 0.1*DR
    :type DR: scalar, optional
    :param kT: tempertature in energy units, defaults to 1.0
    :type kT: scalar, optional
    """    

    def __init__(self,
                 N: int,
                 state_functional,
                 shape:SuperEllipse = _default_sphere,
                 electrodes:Electrodes = _default_qpole,
                 dt:float = 1e-3,
                 DT:float = 0.25,
                 DR:float = 0.025,
                 kT:float = 1.0):
        """
        Constructor method
        """
        super().__init__()
        self._N = N

        assert type(state_functional) == type(lambda _:0), "state functional must be a function object"
        self._lambda_f = state_functional

        # set electrode configuration and reset simulation based on that geometry
        self._elec = electrodes

        self._kT = kT
        self._DT = DT
        self._DR = DR
        self._dt = dt

        # set shape and generate vertices if they're not alrady provided
        self._s = shape
        self._is_disc = np.round(shape.aspect,4)==1.0 and np.round(shape.n,4)==2.0
        if not self._is_disc:
            if not hasattr(shape,'vertices'):
                self._s.contact_vertices(n_verts=16)
            self._Uij = hoomd_alj(self._s,0.0)
        else:
            self._elec.k_rot = np.zeros_like(self._elec.k_rot)
            self._Uij = hoomd_wca(1.0,0.0)
        
        self._alt_methods = None

    @property
    def frame(self) -> gsd.hoomd.Frame:
        """
        :return: the current simulation snapshot which contains particle position/orientation data
        :rtype: `gsd.hoomd.Frame <https://gsd.readthedocs.io/en/stable/python-module-gsd.hoomd.html#gsd.hoomd.Frame>`_
        """        
        if not hasattr(self,'sim'): raise Exception("reset simulation before querying snapshot")
        return self.sim.state.get_snapshot()

    @property
    def num_particles(self) -> int:
        """
        :return: particle count
        :rtype: int
        """        
        return self._N

    @property
    def L(self) -> float:
        """
        :return: simulation box size
        :rtype: scalar
        """        
        return self.frame.configuration.box[0]
    
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
            self._s.contact_vertices(n_verts=16)

    @property
    def interaction(self) -> hoomd.md.pair.Pair:
        """
        :return: the current particle-particle interaction used in the simulation, which is a `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/pair.html>`_ object
        :rtype: `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/pair.html>`_
        """
        return self._Uij

    @interaction.setter
    def interaction(self, pair_potential:hoomd.md.pair.Pair):
        """
        :param pair_potential: the particle-particle interaction to be used in the simulation, which must be a `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/pair.html>`_ object
        :type pair_potential: `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/pair.html>`_
        """
        self._Uij = pair_potential

    @property
    def methods(self) -> list[hoomd.md.methods.Method]:
        """
        :return: the current integration method used in the simulation, which is a `hoomd.md.methods.Method <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/methods/method.html>`_ object and defaults to Brownian Dynamics with this object's current settings for translational and rotational diffusivity.
        :rtype: a list of `hoomd.md.methods.Method <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/methods/method.html>`_ objects
        """
        if self._alt_methods is None:
            methods = [hoomd.md.methods.Brownian(filter=hoomd.filter.All(),
                                                 default_gamma   =  self._kT/self._DT,
                                                 default_gamma_r = [self._kT/self._DR]*3,
                                                 kT=self._kT)]
        elif isinstance(self._alt_methods,hoomd.md.methods.Method):
            methods = [self._alt_methods]
        elif isinstance(self._alt_methods,list) and all([isinstance(m,hoomd.md.methods.Method) for m in self._alt_methods]):
            methods = self._alt_methods
        else:
            raise Exception("methods must be a hoomd.md.methods.Method object or a list of such objects")
            
        return methods

    @methods.setter
    def methods(self, methods:hoomd.md.methods.Method|list[hoomd.md.methods.Method]):
        """
        :param methods: the particle integration methods to be used in the simulation
        :type methods: `hoomd.md.methods.Method <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/methods/method.html>`_ or a list thereof
        """
        self._alt_methods = methods

    @property
    def ideal(self) -> bool:
        """
        :return: whether the simulation is "ideal", meaning there are no particle-particle interactions
        :rtype: bool
        """
        rcuts = np.array([self._Uij.r_cut[k] for k in self._Uij.params.keys()])
        return np.all(rcuts==0)

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
    def electrodes(self) -> Electrodes:
        """
        :return: the electrode configuration which generates the driving force on the particles
        :rtype: :py:class:`Electrodes <utils.hoomd_helpers.Electrodes>` object
        """
        return self._elec
    
    @electrodes.setter
    def electrodes(self, npole:Electrodes):
        """
        :param npole: the electrode configuration which generates the driving force on the particles
        :type npole: :py:class:`Electrodes <utils.hoomd_helpers.Electrodes>` object
        """
        self._elec = npole
    
    @property
    def state(self) -> tuple:
        """applies the state functional to the current simulation particle configuration (including orientations)

        :return: the vector of order parameters, as a tuple, computed by the state functional
        :rtype: tuple
        """
        frame = self.frame
        pts = frame.particles.position
        os = frame.particles.orientation
        return self._lambda_f(pts,os,self._shape)

    def reset(self,
              init_state:gsd.hoomd.Frame | None = None,
              outfile:str | None = None,
              nsnap:float = 0.1,
              seed: int | None = None,
              ):
        """Resets the simulation by creating a new hoomd.Simulation object, reinitializing the configuration using either a user-defined initial state, and redefines the output file if desired. If an output file is specified, the simulation *appends* to the file, meaning consistent specification of the initial state as the last frame of the output file allows for a continuous simulation between :py:meth:`reset()` calls.

        :param init_state: A configuration of particles in the form of a gsd.hoomd.Frame object. Defaults to a random state generated using :py:func:`utils.hoomd_helpers.random_frame`.
        :type init_state: gsd.hoomd.Frame, optional
        :param outfile: a path to an output gsd file to record particle configurations, if none is given the simulation will not save any configurations, defaults to None
        :type outfile: str | None, optional
        :param nsnap: snapshot period in simulation units with which to record the system state to an output gsd file, defaults to 0.1
        :type nsnap: scalar, optional
        :param seed: RNG seed to reinitialize the simulation, defaults to None
        :type seed: int | None, optional
        """        
        super().reset()
        self._time = 0
        
        #remove all writers to close out current gsd file so that subsequent steps continue to append frames.
        if hasattr(self,'sim'):
            for op in self.sim.operations:
                self.sim.operations.remove(op)
        
        #load initial state from frame object into new simulation object
        if seed is None: seed = int(1000*np.random.rand())
        self.sim = hoomd.Simulation(device=hoomd.device.CPU(),seed=seed)
    

        if init_state is None:
            init = random_frame(self._N,3*self._elec.electrode_gap,shape=self._s)
        else:
            init = gsd.hoomd.Frame()
            init.configuration.box = init_state.configuration.box
            init.particles.N = init_state.particles.N
            self._N = init.particles.N

            init.particles.position = init_state.particles.position
            init.particles.orientation = init_state.particles.orientation
            init.particles.typeid = init_state.particles.typeid
            init.particles.types = init_state.particles.types
            init.particles.image = np.zeros_like(init.particles.position)
            init.particles.moment_inertia = init_state.particles.moment_inertia

        self.sim.create_state_from_snapshot(init)
        
        #define file writer which continually appends simulation bursts to a trajectory file
        if not (outfile is None):
            gsd_writer = hoomd.write.GSD(filename=outfile,
                                    trigger=hoomd.trigger.Periodic(int(nsnap/self._dt)),
                                    mode='ab',
                                    dynamic=['property','momentum','attribute'])
            self.sim.operations.writers.append(gsd_writer)

    def run(self,
            time:float,
            k_trans:list|np.ndarray,
            k_rot:list|np.ndarray):
        """
        Steps the simulation forward for a short burst under a harmonic trap.

        :param time: the runtime in simulation units of this short simulation burst
        :type time: scalar
        :param k_trans: a list of field strengths in kT units, applies each along the directions held within the :py:meth:`self.electrodes.direct` attribute.
        :type k_trans: array-like
        :param k_rot: a list of field strengths in kT units, applies each along the directions held within the :py:meth:`self.electrodes.direct` attribute.
        :type k_rot: array-like
        """
        super().run(time,k_trans)
        if not hasattr(self,'sim'): raise Exception("reset simulation before running")

        self._time+=time

        # create extermal potential
        assert len(k_trans) == len(k_rot) == self._elec.num_fields, "the number of translaton/rotational harmonic constants must match the number of harmonic traps"
        npole_args = self._elec.make_npole_BD(k_trans=np.array(k_trans), k_rot = np.array(k_rot)*(not self._is_disc))
        
        forces = [self._Uij]
        for f,t in npole_args:
            npole = hoomd.md.force.Active(filter = hoomd.filter.All())
            npole.active_force['A']  = f
            npole.active_torque['A'] = t
            forces.append(npole)
        
        # instantiate BD stepper with the forces defined above
        integrator = hoomd.md.Integrator(dt=self.dt, methods = self.methods, forces=forces, integrate_rotational_dof = not self._is_disc)
        
        # add field strength to logger so that this quantity is associated with each frame
        if len(self.sim.operations.writers)>0:
            logger = self._elec.make_logger()
            if not self._is_disc: logger.add(self._Uij,quantities=['type_shapes'])
            gsd_writer = self.sim.operations.writers[0]
            gsd_writer.logger = logger
            
        # apply forces to integrator and run for one simstep
        self.sim.operations.integrator = integrator
        simstep = int(time/self._dt)
        self.sim.run(simstep)
        if len(self.sim.operations.writers)>0:
            self.sim.operations.writers[0].flush()
        self.sim.operations.integrator = None




class Quadrupole(Multipole):
    """
    A class for 2D Brownian Dynamics simulations of nearly hard discs in a quadrupolar electrode. 

    Quadrupole simulations are for discs. Therefore the :code:`state_functional` argument of the constructor must be a python function which takes as arguments only the particle positions (Nx3 array).

    :param N: the number of particles in the simulation, which may be modified with :py:meth:`reset()`
    :type N: int
    :param state_functional: A function of which reads the simulation microstate (positions) and returns the desired vector of order parameters as a tuple.
    :param dt: timestep for the brownian integrator, defaults to 2e-4
    :type dt: scalar, optional
    :param DT: Particles' short-time translational diffusivity (in simulation units), defaults to 0.25 which produces a 1-1 MSD in 2D
    :type DT: scalar, optional
    :param DR: Particles' short-time rotational diffusivity (in simulation units), defaults to 0.1*DR
    :type DR: scalar, optional
    :param kT: tempertature in energy units, defaults to 1.0
    :type kT: scalar, optional
    :param diameter: particle diameter in simulation units, defaults to 1.0
    :type diameter: scalar, optional
    :param dg: electrode gap in simulation units, defaults to 30.0
    :type dg: scalar, optional
    :param energy_scale: the energy scale of the particle-particle interaction, defaults to 21.0 for nearly hard discs with a WCA interaction
    :type energy_scale: scalar, optional
    """    

    def __init__(self,
                 N: int,
                 state_functional,
                 dt:float = 1e-3,
                 DT:float = 0.25,
                 DR:float = 0.025,
                 kT:float = 1.0,
                 diameter:float = 1.0,
                 dg:float = 30.0,
                 energy_scale:float = 21.0):
        """
        Constructor method
        """
        ptcl = SuperEllipse(ax=diameter/2,ay=diameter/2)
        qpole = Electrodes(n=2,dg=dg) # WIP electrodes object should get very large dg to keep field actually harmonic, istead of going to tan(x/dg) near edges
        qpole.direct = np.array([0,np.pi/2])
        super().__init__(N,state_functional,shape=ptcl,electrodes=qpole,DT=DT,DR=DR,dt=dt,kT=kT)

        self._2a = diameter
        self._is_disc = True
        self._Uij = hoomd_wca(diameter,energy_scale)
        
    @property
    def diameter(self) -> float:
        """
        :return: the diameter of the discs in simulation units
        :rtype: scalar
        """        
        return self._2a

    @diameter.setter
    def diameter(self, diameter:float):
        """
        :param diameter: the diameter of the discs in simulation units
        :type diameter: float
        """
        self.shape = SuperEllipse(ax=diameter/2,ay=diameter/2)
        self._2a = diameter
    
    @property
    def state(self) -> tuple:
        """applies the state functional to the current simulation particle configuration

        :return: the vector of order parameters, as a tuple, computed by the state functional
        :rtype: tuple
        """        
        pts = self.frame.particles.position
        return self._lambda_f(pts)


    def run(self,time:float,k:float):
        """
        Steps the simulation forward for a short burst under a harmonic trap.

        :param time: the runtime in simulation units of this short simulation burst
        :type time: scalar
        :param k: the field strength in kT units.
        :type k: scalar
        """
        super().run(time,[k,k],[0,0])

    def in_box(self) -> bool:
        """
        :return: whether any particle has exited the main simulation box
        :rtype: bool
        """
        return np.all(self.frame.particles.image[:,:2]==0)




class Octopole(Multipole):
    """
    A class for 2D Brownian Dynamics simulations of nearly hard discs in an octopolar electrode. 

    Octopole simulations are for discs. Therefore the :code:`state_functional` argument of the constructor must be a python function which takes as arguments only the particle positions (Nx3 array).

    :param N: the number of particles in the simulation, which may be modified with :py:meth:`reset()`
    :type N: int
    :param state_functional: A function of which reads the simulation microstate (positions) and returns the desired vector of order parameters as a tuple.
    :param dt: timestep for the brownian integrator, defaults to 2e-4
    :type dt: scalar, optional
    :param DT: Particles' short-time translational diffusivity (in simulation units), defaults to 0.25 which produces a 1-1 MSD in 2D
    :type DT: scalar, optional
    :param DR: Particles' short-time rotational diffusivity (in simulation units), defaults to 0.1*DR
    :type DR: scalar, optional
    :param kT: tempertature in energy units, defaults to 1.0
    :type kT: scalar, optional
    :param diameter: particle diameter in simulation units, defaults to 1.0
    :type diameter: scalar, optional
    :param angles: a pair of directions (in radians) along which to apply harmonic traps, defaults to :math:`\\pm\\pi/4`.
    :type angles: array-like, optional
    :param dg: electrode gap in simulation units, defaults to 30.0
    :type dg: scalar, optional
    :param energy_scale: the energy scale of the particle-particle interaction, defaults to 21.0 for nearly hard discs with a WCA interaction
    :type energy_scale: scalar, optional
    """    

    def __init__(self,
                 N: int,
                 state_functional,
                 dt:float = 1e-3,
                 DT:float = 0.25,
                 DR:float = 0.025,
                 kT:float = 1.0,
                 diameter:float = 1.0,
                 angles:list|tuple|np.ndarray = (np.pi/4,-np.pi/4),
                 dg:float = 30.0,
                 energy_scale:float = 21.0):
        """
        Constructor method
        """
        ptcl = SuperEllipse(ax=diameter/2,ay=diameter/2)
        opole = Electrodes(n=2,dg=dg) # WIP electrodes object should get very large dg to keep field actually harmonic, istead of going to tan(x/dg) near edges
        assert len(angles) == 2, "please supply only two harmonic directions"
        opole.direct = np.array(angles)
        super().__init__(N,state_functional,shape=ptcl,electrodes=opole,DT=DT,DR=DR,dt=dt,kT=kT)

        self._2a = diameter
        self._is_disc = True
        self._Uij = hoomd_wca(diameter,energy_scale)
        
    @property
    def diameter(self) -> float:
        """
        :return: the diameter of the discs in simulation units
        :rtype: scalar
        """        
        return self._2a

    @diameter.setter
    def diameter(self, diameter:float):
        """
        :param diameter: the diameter of the discs in simulation units
        :type diameter: float
        """
        self.shape = SuperEllipse(ax=diameter/2,ay=diameter/2)
        self._2a = diameter

    @property
    def direct(self) -> np.ndarray:
        """
        :return: a pair of directions (in radians) along which to apply harmonic traps (usually at a right angle)
        :rtype: array-like
        """
        return self._elec.direct
    
    @direct.setter
    def direct(self, angles:tuple|list|np.ndarray):
        """
        :param angles: a pair of directions (in radians) along which to apply harmonic traps (usually at a right angle)
        :type angles: array-like
        """
        assert len(angles) == 2, "please supply only two harmonic directions"
        self._elec.direct = np.array(angles)
    
    @property
    def state(self) -> tuple:
        """applies the state functional to the current simulation particle configuration

        :return: the vector of order parameters, as a tuple, computed by the state functional
        :rtype: tuple
        """        
        pts = self.frame.particles.position
        return self._lambda_f(pts)


    def run(self,time:float,k1:float,k2:float):
        """
        Steps the simulation forward for a short burst under a harmonic trap.

        :param time: the runtime in simulation units of this short simulation burst
        :type time: scalar
        :param k1: the field strength along the first harmonic trap axis in kT units.
        :type k1: scalar
        :param k2: the field strength along the second harmonic trap axis in kT units.
        :type k2: scalar
        """
        super().run(time,[k1,k2],[0,0])

    def in_box(self) -> bool:
        """
        :return: whether any particle has exited the main simulation box
        :rtype: bool
        """
        return np.all(self.frame.particles.image[:,:2]==0)


class Coplanar(Multipole):
    """
    A class for 2D Brownian Dynamics simulations of nearly hard colloids between coplanar electrodes. The class takes a superellipse long and short axes (a\\ :sub:`x`, a\\ :sub:`y`) and the superellipse paramter (n) as optional parameters in the constructor and makes them settable properties. Additionally, :py:class:`Coplanar` inherits :py:class:`Multipole`, meaning the :code:`shape` property is also settable with any :py:class:`SuperEllipse <utils.geometry.SuperEllipse>` object.

    Coplanar simulations are for generic shapes. Therefore the :code:`state_functional` parameter of the constructor must be a python function which takes as arguments the particle positions (Nx3 array), the quaternion orientations of the particles (Nx4 array) (see :py:meth:`quat_to_angle <utils.geometry.quat_to_angle>` to convert), and the simulation's :code:`shape` property (:py:class:`SuperEllipse <utils.geometry.SuperEllipse>`). For anisotropic particles specifically, we approximate hard-particle behavior using a `hoomd.md.pair.aniso.ALJ <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/aniso/alj.html>`_ class.

    :param N: the number of particles in the simulation, which may be modified with :py:meth:`reset()`
    :type N: int
    :param state_functional: A function of which reads the simulation microstate (positions, orientations) and returns the desired vector of order parameters as a tuple.
    :param dt: timestep for the brownian integrator, defaults to 2e-4
    :type dt: scalar, optional
    :param DT: Particles' short-time translational diffusivity (in simulation units), defaults to 0.25 which produces a 1-1 MSD in 2D
    :type DT: scalar, optional
    :param DR: Particles' short-time rotational diffusivity (in simulation units), defaults to 0.1*DR
    :type DR: scalar, optional
    :param kT: tempertature in energy units, defaults to 1.0
    :type kT: scalar, optional
    :param ax: the particle's long axis in simulation units, defaults to 1.0
    :type ax: float, optional
    :param ay: the particle's short axis in simulation units, defaults to 0.5
    :type ay: float, optional
    :param superellipse_param: the particle's superellipse parameter, defaults to 2.0
    :type superellipse_param: float, optional
    :param dg: electrode gap in simulation units, defaults to 30.0
    :type dg: scalar, optional
    :param energy_scale: the energy scale (in kT units) of the contact-contact interactions within the ALJ framework, defaults to 1.0
    :type energy_scale: scalar, optional
    :param contact_radius: the contact radius of the ALJ interaction centers in simulation units, defaults to 0.1
    :type contact_radius: scalar, optional
    :param n_verts: the number of vertices per superelliptical particle, defaults to 16
    :type n_verts: int, optional
    :param require_corners: whether to force vertices to be placed at the corners of the superellipse, defaults to False
    :type require_corners: bool, optional
    """

    def __init__(self,
                 N: int,
                 state_functional,
                 dt:float = 1e-4,
                 DT:float = 0.25,
                 DR:float = 0.025,
                 kT:float = 1.0,
                 ax:float = 1.0, ay:float=0.5, superellipse_param:float=2.0,
                 dg:float = 30.0,
                 energy_scale:float   = 1.0,
                 contact_radius:float = 0.1,
                 n_verts:int = 16,
                 require_corners:bool = False,
                 ):
        """
        Constructor method
        """
        ptcl = SuperEllipse(ax=ax,ay=ay,n=superellipse_param)
        ptcl.contact_vertices(n_verts=n_verts,contact_ratio=contact_radius/(ptcl.ay),require_corners=require_corners)
        bpole = Electrodes(n=1,dg=dg)
        bpole.direct = np.array([0])
        super().__init__(N,state_functional,shape=ptcl,electrodes=bpole,DT=DT,DR=DR,dt=dt,kT=kT)
        self._Uij = hoomd_alj(ptcl,energy_scale)

    @property
    def major_axis(self) -> float:
        """
        :return: the particle's long axis in simulation units
        :rtype: float
        """
        return self._s.ax

    @major_axis.setter
    def major_axis(self, ax:float):
        """
        :param ax: the particle's long axis in simulation units
        :type ax: float
        """
        ptcl = SuperEllipse(ax=ax,ay=self._s.ay,n=self._s.n)
        self.shape = ptcl

    @property
    def minor_axis(self) -> float:
        """
        :return: the particle's short axis in simulation units
        :rtype: float
        """
        return self._s.ay

    @minor_axis.setter
    def minor_axis(self, ay:float):
        """
        :param ax: the particle's short axis in simulation units
        :type ax: float
        """
        ptcl = SuperEllipse(ax=self._s.ax,ay=ay,n=self._s.n)
        self.shape = ptcl

    @property
    def superellipse_param(self) -> float:
        """
        :return: the particle's short axis in simulation units
        :rtype: float
        """
        return self._s.ay

    @superellipse_param.setter
    def superellipse_param(self, n:float):
        """
        :param n: the particle's superellipse parameter (corner sharpness)
        :type ax: float
        """
        ptcl = SuperEllipse(ax=self._s.ax,ay=self._s.ay,n=n)
        self.shape = ptcl

    @property
    def state(self) -> tuple:
        """applies the state functional to the current simulation particle configuration (including orientations)

        :return: the vector of order parameters, as a tuple, computed by the state functional
        :rtype: tuple
        """
        frame = self.frame
        pts = frame.particles.position
        os = frame.particles.orientation
        return self._lambda_f(pts,os,self.shape)
    
    def run(self,time:float,k_trans:float,k_rot:float):
        """
        Steps the simulation forward for a short burst under a harmonic trap.

        :param time: the runtime in simulation units of this short simulation burst
        :type time: scalar
        :param k_trans: the translational field strength in kT units
        :type k_trans: scalar
        :param k_rot: the rotational field strength in kT units
        :type k_rot: scalar
        """
        super().run(time,[k_trans],[k_rot])

    def in_box(self) -> bool:
        """
        :return: whether any particle has exited the main simulation box
        :rtype: bool
        """
        return np.all(self.frame.particles.image[:,0]==0)