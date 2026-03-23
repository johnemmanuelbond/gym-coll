# -*- coding: utf-8 -*-
"""
A group of simuation classes meant for simulating quasi-2D ensembles of hard colloidal particles within various electrode geometries. The main :py:class:`Multipole` class can hanle any particle geometry and any electrode geometry, but there are also subclasses designed to simulate specific experimental cases, spheres in a quadrupole, spheres in an octopole, and superellipses (ellipses/rectangles) between coplanar electrodes. Each of these classes is written to function in common 'simulation units' which is a nondimensionalization scheme to report energies, and lengths without referring to experimental conditions.

- Energy units: the thermal energy scale :math:`kT`
- Length units: the diameter of spherical colloids :math:`l=2a`, or the short axis of anisotropic colloids :math:`l=2a_y`.

Monte Carlo is adept at simulating "hard" colloidal particles which interact only by excluded volume. So, this class makes heavy use of the :py:class:`SuperEllipse <utils.geometry.SuperEllipse>` class to represent numerous particle geometries including spheres, ellipses, rectangles, and rhombuses.

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
from utils import SuperEllipse, Electrodes, random_frame, hpmc_dipoles

_default_sphere = SuperEllipse(ax=0.5,ay=0.5)
_default_qpole = Electrodes(n=2,dg=30)
_default_qpole.direct = np.array([0,np.pi/2])

class Multipole(Simbase):
    """
    A generic class for 2D Monte Carlo simulations of shaped hard particles in any multipolar electrode configuration.

    Multipole simulations are for generic shapes. Therefore the :code:`state_functional` parameter of the constructor must be a python function which takes as arguments the particle positions (Nx3 array), the quaternion orientations of the particles (Nx4 array) (see :py:meth:`quat_to_angle <utils.geometry.quat_to_angle>` to convert), and the simulation's :code:`shape` property (:py:class:`SuperEllipse <utils.geometry.SuperEllipse>`).

    :param N: the number of particles in the simulation, which may be modified with :py:meth:`reset()`
    :type N: int
    :param state_functional: A function of which reads the simulation microstate (positions, orientations) and returns the desired vector of order parameters as a tuple.
    :param shape: a particle shape, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>` object, optional
    :param electrodes: en electrode configuration, defaults to a symmetric quadrupole
    :type electrodes: :py:class:`Electrodes <utils.hoomd_helpers.Electrodes>`, optional
    """    

    def __init__(self,
                 N: int,
                 state_functional,
                 shape:SuperEllipse = _default_sphere,
                 electrodes:Electrodes = _default_qpole):
        """
        Constructor method
        """
        super().__init__()
        self._N = N

        assert type(state_functional) == type(lambda _:0), "state functional must be a function object"
        self._lambda_f = state_functional

        # set electrode configuration and reset simulation based on that geometry
        self._elec = electrodes

        self.ideal = False

        # set monte carlo move sizes to sensible defaults
        self._dx = shape.ay/10
        self._da = self._dx*(0.03)**0.5

        # set shape and generate vertices if they're not alrady provided
        self._s = shape
        self._is_disc = np.round(shape.aspect,4)==1.0 and np.round(shape.n,4)==2.0
        self._use_dipoles = False
        self._dipole_energy = 0.0
        if not self._is_disc:
            if not hasattr(shape,'vertices'):
                self._s.contact_vertices()
        else:
            self._elec.k_rot = np.zeros_like(self._elec.k_rot)


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
            self._s.contact_vertices()

    @property
    def elapsed(self)-> int:
        """
        :return: the total number of MC sweeps since the last reset
        :rtype: int
        """
        return self._sweeps
    
    @property
    def dx(self) -> float:
        """
        :return: maximum translation (in simulation units) per particle per sweep
        :rtype: float
        """        
        return self._dx

    @dx.setter
    def dx(self,stepsize:float):
        """
        :param stepsize: maximum translation (in simulation units) per particle per sweep
        :type stepsize: float
        """        
        self._dx = stepsize

    @property
    def da(self) -> float:
        """
        :return: maximum rotation (in radians) per particle per sweep
        :rtype: float
        """        
        return self._da

    @da.setter
    def da(self,stepsize:float):
        """
        :param stepsize: maximum rotation (in radians) per particle per sweep
        :type stepsize: float
        """        
        self._da = stepsize
    
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
    def dipole_energy(self) -> float|None:
        if not self._use_dipoles: return None
        else: return self._dipole_energy
    
    @dipole_energy.setter
    def dipole_energy(self, energy:float):
        if energy == 0.0 or self._is_disc:
            self._use_dipoles = False
            self._dipole_energy = 0.0
            self._udd = []
        else:
            self._use_dipoles = True
            self._dipole_energy = energy
            d1, d2 = hpmc_dipoles(self.shape, self._dipole_energy)
            self._udd = [d1,d2]
        
    
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

    def reset(self,
              init_state:gsd.hoomd.Frame | None = None,
              outfile:str | None = None,
              nsnap:int = 100,
              seed: int | None = None,
              ):
        """Resets the simulation by creating a new hoomd.Simulation object, reinitializing the configuration using either a user-defined initial state, and redefines the output file if desired. If an output file is specified, the simulation *appends* to the file, meaning consistent specification of the initial state as the last frame of the output file allows for a continuous simulation between :py:meth:`reset()` calls.

        :param init_state: A configuration of particles in the form of a gsd.hoomd.Frame object. Defaults to a random state generated using :py:func:`utils.hoomd_helpers.random_frame`.
        :type init_state: gsd.hoomd.Frame, optional
        :param outfile: a path to an output gsd file to record particle configurations, if none is given the simulation will not save any configurations, defaults to None
        :type outfile: str | None, optional
        :param nsnap: snapshot period (of MC sweeps) with which to record the system state to an output gsd file, defaults to 100
        :type nsnap: int, optional
        :param seed: RNG seed to reinitialize the simulation, defaults to None
        :type seed: int | None, optional
        """        
        super().reset()
        self._sweeps = 0
        
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
                                    trigger=hoomd.trigger.Periodic(nsnap),
                                    mode='ab',
                                    dynamic=['property','momentum','attribute'])
            self.sim.operations.writers.append(gsd_writer)

    def run(self,
            sweeps:int,
            k_trans:list|np.ndarray,
            k_rot:list|np.ndarray):
        """
        Steps the simulation forward for a short burst under a harmonic trap.

        :param sweeps: the number of monte carlo sweeps (move attempts per particle) of this short simulation burst
        :type sweeps: int
        :param k_trans: a list of field strengths in kT units, applies each along the directions held within the :py:meth:`self.electrodes.direct` attribute.
        :type k_trans: array-like
        :param k_rot: a list of field strengths in kT units, applies each along the directions held within the :py:meth:`self.electrodes.direct` attribute.
        :type k_rot: array-like
        """
        super().run(sweeps,k_trans)
        if not hasattr(self,'sim'): raise Exception("reset simulation before running")

        self._sweeps+=sweeps

        # set up hard particle monte carlo integrator
        if self._is_disc:
            mc = hoomd.hpmc.integrate.Sphere(nselect=2, translation_move_probability=1.0,default_d=self._dx)
            mc.shape["A"] = dict(diameter=2*self._s.ay)
        else:
            mc = hoomd.hpmc.integrate.ConvexSpheropolygon(nselect=3,translation_move_probability=2/3,default_a=self._da,default_d=self._dx)
            mc.shape["A"] = dict(
                vertices = self._s.vertices[:,:2].tolist(),
                sweep_radius = self._s.contact_ratio*self._s.ay,
            )
        if self.ideal: mc.interaction_matrix['A','A'] = False
        
        # set up dipole interactions if requested by setting 'dipole_energy'
        if self._use_dipoles: mc.pair_potentials = self._udd

        # create extermal potential
        assert len(k_trans) == len(k_rot) == self._elec.num_fields, "the number of translaton/rotational harmonic constants must match the number of harmonic traps"
        npole_args = self._elec.make_npole_MC(k_trans=np.array(k_trans),k_rot=np.array(k_rot)*(not self._is_disc),pnum=self._N)
        mc.external_potentials = [hoomd.hpmc.external.Harmonic(**a) for a in npole_args]
        
        # add field strength to logger so that this quantity is associated with each frame
        if len(self.sim.operations.writers)>0:
            logger = self._elec.make_logger()
            if not self._is_disc: logger.add(mc,quantities=['type_shapes'])
            gsd_writer = self.sim.operations.writers[0]
            gsd_writer.logger = logger
            
        # apply forces to integrator and run for one simstep
        self.sim.operations.integrator = mc
        self.sim.run(sweeps)
        if len(self.sim.operations.writers)>0:
            self.sim.operations.writers[0].flush()
        self.sim.operations.integrator = None




class Quadrupole(Multipole):
    """
    A class for 2D Monte Carlo simulations of hard discs in a quadrupolar electrode. 

    Quadrupole simulations are for discs. Therefore the :code:`state_functional` argument of the constructor must be a python function which takes as arguments only the particle positions (Nx3 array).

    :param N: the number of particles in the simulation, which may be modified with :py:meth:`reset()`
    :type N: int
    :param state_functional: A function of which reads the simulation microstate (positions) and returns the desired vector of order parameters as a tuple.
    :param diameter: particle diameter in simulation units, defaults to 1.0
    :type diameter: scalar, optional
    :param dg: electrode gap in simulation units, defaults to 30.0
    :type dg: scalar, optional
    """    

    def __init__(self,
                 N: int,
                 state_functional,
                 diameter:float = 1.0,
                 dg:float = 30.0):
        """
        Constructor method
        """
        ptcl = SuperEllipse(ax=diameter/2,ay=diameter/2)
        qpole = Electrodes(n=2,dg=dg) # WIP electrodes object should get very large dg to keep field actually harmonic, istead of going to tan(x/dg) near edges
        qpole.direct = np.array([0,np.pi/2])
        super().__init__(N,state_functional,shape=ptcl,electrodes=qpole)

        self._2a = diameter
        self._is_disc = True
        
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
        self._2a = diameter
        self.shape = SuperEllipse(ax=diameter/2,ay=diameter/2)
    
    @property
    def state(self) -> tuple:
        """applies the state functional to the current simulation particle configuration

        :return: the vector of order parameters, as a tuple, computed by the state functional
        :rtype: tuple
        """        
        pts = self.frame.particles.position
        return self._lambda_f(pts)


    def run(self,sweeps:int,k:float):
        """
        Steps the simulation forward for a short burst under a harmonic trap.

        :param sweeps: the number of monte carlo sweeps (move attempts per particle) of this short simulation burst
        :type sweeps: int
        :param k: the field strength in kT units.
        :type k: scalar
        """
        super().run(sweeps,[k,k],[0,0])

    def in_box(self) -> bool:
        """
        :return: whether any particle has exited the main simulation box
        :rtype: bool
        """
        return np.all(self.frame.particles.image[:,:2]==0)




class Octopole(Multipole):
    """
    A class for 2D Monte Carlo simulations of hard discs in an octopolar electrode. 

    Octopole simulations are for discs. Therefore the :code:`state_functional` argument of the constructor must be a python function which takes as arguments only the particle positions (Nx3 array).

    :param N: the number of particles in the simulation, which may be modified with :py:meth:`reset()`
    :type N: int
    :param state_functional: A function of which reads the simulation microstate (positions) and returns the desired vector of order parameters as a tuple.
    :param diameter: particle diameter in simulation units, defaults to 1.0
    :type diameter: scalar, optional
    :param angles: a pair of directions (in radians) along which to apply harmonic traps, defaults to :math:`\\pm\\pi/4`.
    :type angles: array-like, optional
    :param dg: electrode gap in simulation units, defaults to 30.0
    :type dg: scalar, optional
    """    

    def __init__(self,
                 N: int,
                 state_functional,
                 diameter:float = 1.0,
                 angles:list|tuple|np.ndarray = (np.pi/4,-np.pi/4),
                 dg:float = 30.0):
        """
        Constructor method
        """
        ptcl = SuperEllipse(ax=diameter/2,ay=diameter/2)
        opole = Electrodes(n=2,dg=dg) # WIP electrodes object should get very large dg to keep field actually harmonic, istead of going to tan(x/dg) near edges
        assert len(angles) == 2, "please supply only two harmonic directions"
        opole.direct = np.array(angles)
        super().__init__(N,state_functional,shape=ptcl,electrodes=opole)

        self._2a = diameter
        self._is_disc = True
        
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
        self._2a = diameter
        self.shape = SuperEllipse(ax=diameter/2,ay=diameter/2)

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


    def run(self,sweeps:int,k1:float,k2:float):
        """
        Steps the simulation forward for a short burst under a harmonic trap.

        :param sweeps: the number of monte carlo sweeps (move attempts per particle) of this short simulation burst
        :type sweeps: int
        :param k1: the field strength along the first harmonic trap axis in kT units.
        :type k1: scalar
        :param k2: the field strength along the second harmonic trap axis in kT units.
        :type k2: scalar
        """
        super().run(sweeps,[k1,k2],[0,0])

    def in_box(self) -> bool:
        """
        :return: whether any particle has exited the main simulation box
        :rtype: bool
        """
        return np.all(self.frame.particles.image[:,:2]==0)


class Coplanar(Multipole):
    """
    A class for 2D Monte Carlo simulations of hard colloids between coplanar electrodes. The class takes a superellipse long and short axes (a\\ :sub:`x`, a\\ :sub:`y`) and the superellipse paramter (n) as optional parameters in the constructor and makes them settable properties. Additionally, :py:class:`Coplanar` inherits :py:class:`Multipole`, meaning the :code:`shape` property is also settable with any :py:class:`SuperEllipse <utils.geometry.SuperEllipse>` object.

    Coplanar simulations are for generic shapes. Therefore the :code:`state_functional` parameter of the constructor must be a python function which takes as arguments the particle positions (Nx3 array), the quaternion orientations of the particles (Nx4 array) (see :py:meth:`quat_to_angle <utils.geometry.quat_to_angle>` to convert), and the simulation's :code:`shape` property (:py:class:`SuperEllipse <utils.geometry.SuperEllipse>`).

    :param N: the number of particles in the simulation, which may be modified with :py:meth:`reset()`
    :type N: int
    :param state_functional: A function of which reads the simulation microstate (positions, orientations) and returns the desired vector of order parameters as a tuple.
    :param ax: the particle's long axis in simulation units, defaults to 1.0
    :type ax: float, optional
    :param ay: the particle's short axis in simulation units, defaults to 0.5
    :type ay: float, optional
    :param superellipse_param: the particle's superellipse parameter, defaults to 2.0
    :type superellipse_param: float, optional
    :param dg: electrode gap in simulation units, defaults to 30.0
    :type dg: scalar, optional
    """

    def __init__(self,
                 N: int,
                 state_functional,
                 ax:float = 1.0, ay:float=0.5, superellipse_param:float=2.0,
                 dg:float = 30.0):
        """
        Constructor method
        """
        ptcl = SuperEllipse(ax=ax,ay=ay,n=superellipse_param)
        bpole = Electrodes(n=1,dg=dg)
        bpole.direct = np.array([0])
        super().__init__(N,state_functional,shape=ptcl,electrodes=bpole)
    
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
    
    def run(self,sweeps:int,k_trans:float,k_rot:float):
        """
        Steps the simulation forward for a short burst under a harmonic trap.

        :param sweeps: the number of monte carlo sweeps (move attempts per particle) of this short simulation burst
        :type sweeps: int
        :param k_trans: the translational field strength in kT units
        :type k_trans: scalar
        :param k_rot: the rotational field strength in kT units
        :type k_rot: scalar
        """
        super().run(sweeps,[k_trans],[k_rot])

    def in_box(self) -> bool:
        """
        :return: whether any particle has exited the main simulation box
        :rtype: bool
        """
        return np.all(self.frame.particles.image[:,0]==0)