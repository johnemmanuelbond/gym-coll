# -*- coding: utf-8 -*-
"""
Contains a few methods to simplify interacting with `hoomd-blue <https://hoomd-blue.readthedocs.io/en/latest>`_. Includes a class to represent electrode geometries.
"""
import numpy as np
from scipy.spatial.distance import squareform, pdist

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
if has_hoomd: import hoomd

from .geometry import SuperEllipse


def random_frame(N:int, W:float, H:float=None,
                 shape:SuperEllipse=SuperEllipse(ax=0.5,ay=0.5),
                 types:list = ['A'],
                 rng=np.random.default_rng()) -> gsd.hoomd.Frame:
    """
    :param N: number of particles to randomly distribute in the frame
    :type N: int
    :param W: width of the box containing particles
    :type W: float
    :param H: the height of the box containing the particles, defaults to W
    :type H: float, optional
    :param shape: the shape of particles to generate a nonoverlapping configuration for, defaults to a disc with diameter 1.0
    :type shape: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>`, optional
    :param types: a list of particle types to include in the gsd Frame, defaults to ['A']
    :type types: list, optional
    :param rng: _description_, defaults to np.random.default_rng()
    :type rng: Generator, optional
    :return: a random configuration of `N` nonoverlapping particles within a box.
    :rtype: `gsd.hoomd.Frame <https://gsd.readthedocs.io/en/stable/python-module-gsd.hoomd.html#gsd.hoomd.Frame>`_
    """    
    #Prodcues a random configuration of N non-overlapping particles within and W x H box

    # assumes a square box if not otherwise specified
    if H is None: H=W
    if not hasattr(shape,'vertices'):
        shape.contact_vertices()

    # pick a set of initially nonpverlapping points within an WxH box
    pts_init = (rng.random((N,3)) - 0.5)
    pts_init[:,0]*=0.9*(W-shape.outsphere)/shape.ax/2
    pts_init[:,1]*=(H-2*shape.ay)/shape.ay/2
    pts_init[:,2]*=0
    free = squareform(pdist(pts_init)>1.15)
    pts = pts_init[np.all(free,axis=-1)]

    # add particles to the configuration until there are 'N' total particles
    count=0
    while len(pts) < N:
        # do not go indefinitely, if it takes 1000 attempts use a different method to initialize
        assert count < 1000, "could not randomly configure particles within given box"
        count+=1

        #pick a random point and check if it overlapps
        r_pt = (rng.random((1,3)) - 0.5)
        r_pt[:,0]*=0.9*(W-shape.outsphere)/shape.ax/2
        r_pt[:,1]*=(H-2*shape.ay)/shape.ay/2
        r_pt[:,2]*=0
        if np.all(np.linalg.norm(pts-r_pt,axis=-1)>1.15):
            #add to configuration
            pts = np.append(pts,r_pt,axis=0)
            count=0

    pts[:,0]*=2*shape.ax
    pts[:,1]*=2*shape.ay

    # assemble positions into a gsd frame
    frame = gsd.hoomd.Frame()
    frame.configuration.box = [W,H,0,0,0,0]
    frame.particles.N = N
    frame.particles.position = pts
    frame.particles.typeid=[0]*N
    frame.particles.types=types
    frame.particles.image = np.zeros((N,3))

    if np.round(shape.aspect) != 1.0:
        frame.particles.moment_inertia = [[0,0,10]]*N
        thetas = (rng.random(N)-0.5)*2*np.pi/8
        frame.particles.orientation = np.array([np.cos(thetas/2),np.zeros(N),np.zeros(N),np.sin(thetas/2)]).T

    return frame




class Electrodes:
    """
    A class to contain helpful methods for representing electrode geometries, and generating arguments to pass into `hoomd-blue`_ classes. The class represents electrodes as a superposition of nearly harmonic traps. The energy associated with the :math:`i` th one of these traps can be written like:

    .. math::

        U_{{tr,i}} = \\frac{{1}}{{2}}k_{{tr,i}}\\tan^2\\big(r/d_g\\big)\\approx\\frac{{1}}{{2}}k_{{tr,i}}\\big(r/d_g\\big)^2

    .. math::

        U_{{rot,i}} = \\frac{{1}}{{2}}k_{{rot,i}}\\sin^2\\big(m\\Delta\\theta\\big)

    Where :math:`k_{{tr,i}}` and :math:`k_{{rot,i}}` correspond to the :math:`i` th translational and rotational energy scales (in kT units), :math:`d_g` is the distance between the electrodes (in simulation units), and :math:`m` is a symmetry factor corresponding to the particles within the electrode (a rectangle has :math:`m=1`, a square has :math:`m=2`, and a disc has :math:`m=\\infty`). The unitless distance :math:`r` and the angle :math:`\\Delta\\theta` are defined relative to an angle :math:`\\theta_i` which defines the axis along which the harmonic trap drives particle translation, and along which the trap aligns particle orientations:

    .. math::
        
        r \\equiv \\vec{{r_p}}\\cdot\\hat{{d}}/d_g= \\frac{{1}}{{d_g}}\\big(x_p\\cos\\theta_i + y_p\\sin\\theta_i\\big)

    .. math::

        \\Delta\\theta \\equiv \\theta_p - \\theta_i
    
    Where :math:`x_p`, :math:`y_p`, :math:`\\theta_p` are the position and orientation of a particle within the electrodes.

    A generic field configuraiton can be represented as a superposition of these harmonic traps:

    .. math::

        U_{{tr}} = \\sum_i U_{{tr,i}} \\qquad U_{{rot}} = \\sum_i U_{{rot,i}}

    :param n: number of fields to superimpose, defaults to 2
    :type n: int, optional
    :param dg: gap between all sets of electrodes (in simulation units), defaults to 30
    :type dg: float, optional
    """        
    def __init__(self, n:int=2, dg:float=30):
        """
        Constructor
        """
        self.k_trans = np.zeros(n)
        self.k_rot = np.zeros(n)
        self._dg = dg
        self.direct = np.linspace(0,np.pi,n,endpoint=False)

    @property
    def num_fields(self)->int:
        """
        :return: number of fields to superimpose
        :rtype: int
        """        
        return len(self.direct)

    @property
    def electrode_gap(self)->float:
        """
        :return: gap between all sets of electrodes
        :rtype: float
        """        
        return self._dg
    
    @electrode_gap.setter
    def electrode_gap(self, dg:float):
        """
        :param dg: gap between all sets of electrodes
        :type dg: float
        """        
        self._dg = dg

    def U_trans(self,xs:np.ndarray,ys:np.ndarray,
                k_trans:list|np.ndarray = None,
                direct:list|np.ndarray=None)->np.ndarray:
        """Computes the potential energy of a set of x- and y-positions according to the electrode's current configuration.

        :param xs: a set of x-positions (in simulation length units) to calculate energies at
        :type xs: np.ndarray
        :param ys: a set of y-positions (in simulation length units) to calculate energies at
        :type ys: np.ndarray
        :param k_trans: sets the translational field strengths in kT units constraining particles along each multipole axis, defaults to None
        :type k_trans: list | np.ndarray, optional
        :param direct: sets the direction (in radians) of each multipole axis, defaults to None
        :type direct: list | np.ndarray, optional
        :return: a set of potential energies at each x- and y-position 
        :rtype: np.ndarray
        """        
        if not (k_trans is None): self.k_trans = k_trans
        if not (direct is None): self.direct = direct

        n = len(self.direct)
        
        shape = (*(xs.T.shape),n)
        cs = np.full(shape,np.cos(self.direct)).T
        ss = np.full(shape,np.sin(self.direct)).T
        ks = np.full(shape,self.k_trans).T

        shape = (n,*(xs.shape))
        all_xs = np.full(shape,xs)/self._dg
        all_ys = np.full(shape,ys)/self._dg
        rs = all_xs*cs + all_ys*ss

        return np.sum(0.5 * ks * rs**2,axis=0)
    
    def U_rot(self,angles:np.ndarray,
              k_rot:list|np.ndarray = None,
              direct:list|np.ndarray=None,m:int=1)->np.ndarray:
        """Computes potential energy of a set of particle orientations according to the electrode's current configuration.

        :param angles: a set of orientations (in radians) to compute potential energies at
        :type angles: np.ndarray
        :param k_rot: sets the rotational field strengths in kT units aligning particles along each multipole axis, defaults to None
        :type k_rot: list | np.ndarray, optional
        :param direct: sets the direction (in radians) of each multipole axis, defaults to None, defaults to None
        :type direct: list | np.ndarray, optional
        :param m: symmetry factor of the particles, defaults to 1
        :type m: int, optional
        :return: a set of potential energies at each orientation
        :rtype: np.ndarray
        """        
        if not (k_rot is None): self.k_rot = k_rot
        if not (direct is None): self.direct = direct

        n = len(self.direct)

        shape = (*(angles.T.shape),n)
        ks = np.full(shape,self.k_rot).T
        t0s = np.full(shape,self.direct).T
        
        shape = (n,*(angles.shape))
        all_angles = np.full(shape,angles)
        ss = np.sin(m*(all_angles - t0s))

        return np.sum(0.5 * ks * ss**2,axis=0)

    def make_npole_MC(self,pnum:int=1,
                      k_trans:list|np.ndarray=None,
                      k_rot:list|np.ndarray=None,
                      direct:list|np.ndarray=None,m:int=1)->dict:
        """generates the arguments needed to pass the current electrode configuration into hoomd `hpmc <https://hoomd-blue.readthedocs.io/en/latest/hoomd/module-hpmc.html>`_ simulations. To properly use this method, ensure that the linked version of hoomd has the `ExternalFieldHarmonic.h` C++ header modifed as shown in the hoomd-mods/hoomd-v5-npole subdirectory of the SMRL repository.

        :param pnum: number of particles in the hoomd simulation which uses this output (needed to correctly use `hoomd.hpmc.external.Harmonic <https://hoomd-blue.readthedocs.io/en/latest/hoomd/hpmc/external/harmonic.html>`_), defaults to 1
        :type pnum: int, optional
        :param k_trans: sets the translational field strengths in kT units constraining particles along each multipole axis, defaults to None
        :type k_trans: list | np.ndarray, optional
        :param k_rot: sets the rotational field strengths in kT units aligning particles along each multipole axis, defaults to None
        :type k_rot: list | np.ndarray, optional
        :param direct: sets the direction (in radians) of each multipole axis, defaults to None, defaults to None
        :type direct: list | np.ndarray, optional
        :param m: symmetry factor of the particles, defaults to 1
        :type m: int, optional
        :return: a list of dictionaries which correspond to arguments to be passed into the modified `hoomd.hpmc.external.Harmonic <https://hoomd-blue.readthedocs.io/en/latest/hoomd/hpmc/external/harmonic.html>`_
        :rtype: dict
        """        
        if not (k_trans is None): self.k_trans = k_trans
        if not (k_rot is None): self.k_rot = k_rot
        if not (direct is None): self.direct = direct

        sym = np.array([[1,0,0,0]]*int(2*m))
        self.Harmonic = [dict(reference_positions=self._dg*np.full((pnum,3),[np.cos(o),np.sin(o),0]),
                     k_translational=kt,
                     symmetries=sym,
                     reference_orientations=np.full((pnum,4),[np.cos(o/2),0,0,np.sin(o/2)]),
                     k_rotational = kr) for kt,kr,o in zip(self.k_trans.copy(),self.k_rot.copy(),self.direct.copy())]
        return self.Harmonic
    
    def make_npole_BD(self,k_trans:list|np.ndarray=None,
                      k_rot:list|np.ndarray=None,
                      direct:list|np.ndarray=None,m:int=1)->list[tuple[list]]:
        """generates the arguments needed to pass the current electrode configuration into hoomd `md <https://hoomd-blue.readthedocs.io/en/latest/hoomd/module-md.html>`_ simulations. To properly use this method, ensure that the linked version of hoomd has the `ActiveForceCompute.cc` C++ file modifed as shown in the hoomd-mods/hoomd-v5-npole subdirectory of the SMRL repository.

        :param k_trans: sets the translational field strengths in kT units constraining particles along each multipole axis, defaults to None
        :type k_trans: list | np.ndarray, optional
        :param k_rot: sets the rotational field strengths in kT units aligning particles along each multipole axis, defaults to None
        :type k_rot: list | np.ndarray, optional
        :param direct: sets the direction (in radians) of each multipole axis, defaults to None, defaults to None
        :type direct: list | np.ndarray, optional
        :param m: symmetry factor of the particles, defaults to 1
        :type m: int, optional
        :return: a list of pairs of force and torque vectors which correspond to arguments to be passed into the modified `hoomd.md.force.Active <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/force/active.html>`_
        :rtype: list[tuple[list]]
        """        
        if not (k_trans is None): self.k_trans = k_trans
        if not (k_rot is None): self.k_rot = k_rot
        if not (direct is None): self.direct = direct

        self.Active = [([kt,o,self._dg],[kr,o,m]) for kt, kr, o in zip(self.k_trans.copy(),self.k_rot.copy(),self.direct.copy())]
        return self.Active
    
    def make_logger(self,k_trans:list|np.ndarray=None,
                    k_rot:list|np.ndarray=None,
                    direct:list|np.ndarray=None):
        """creates a `logging <https://hoomd-blue.readthedocs.io/en/latest/hoomd/module-logging.html>`_ object so that simulations write the field configuration to `gsd <https://gsd.readthedocs.io/en/stable/index.html>`_ files. This way scripts which read these files can create :py:class:`Electrodes` objects for recreating simulation objects or rendering energy landscapes.

        :param k_trans: sets the translational field strengths in kT units constraining particles along each multipole axis, defaults to None
        :type k_trans: list | np.ndarray, optional
        :param k_rot: sets the rotational field strengths in kT units aligning particles along each multipole axis, defaults to None
        :type k_rot: list | np.ndarray, optional
        :param direct: sets the direction (in radians) of each multipole axis, defaults to None, defaults to None
        :type direct: list | np.ndarray, optional
        :return: a hoomd `logging <https://hoomd-blue.readthedocs.io/en/latest/hoomd/module-logging.html>`_ object to record the field configuration
        :rtype: `hoomd.logging.Logger <https://hoomd-blue.readthedocs.io/en/latest/hoomd/logging/logger.html#>`_
        """        
        if not (k_trans is None): self.k_trans = k_trans
        if not (k_rot is None): self.k_rot = k_rot
        if not (direct is None): self.direct = direct
    
        action_log = hoomd.logging.Logger(only_default=False)
        action_log[('k_trans')] = (lambda: self.k_trans.copy(), 'scalar')
        action_log[('k_rot')] = (lambda: self.k_rot.copy(), 'scalar')
        action_log[('direct')] = (lambda: self.direct.copy(), 'scalar')
        action_log[('dg')] = (lambda: [self._dg], 'scalar')
        return action_log


def hoomd_dlvo(debye_length:float, energy_scale:float, buffer_size:float=0.4):
    """
    Creates a `hoomd.md.pair.DLVO <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/dlvo.html>`_ object with the given debye length and energy scale. The DLVO interaction is a screened electrostatic potential for simulating charged colloids in an electrolyte. This interation has the form:

    .. math::
        U(r) = A e^{-\\kappa (r-2a)}

    Where :math:`\\kappa=2a/\\lambda_D` where :math:`\\lambda_D` is the debye kength of this screened electrostatic repulsion

    :param debye_length: The debye length in simulation units
    :type debye_length: float
    :param energy_scale: The prefactor on the exponential energy scale of the interaction in kT units
    :type energy_scale: float
    :param buffer_size: The buffer size for the neighbor list, defaults to 0.4
    :type buffer_size: float, optional
    :return: A hoomd `md.pair.DLVO` object
    :rtype: hoomd.md.pair.Pair
    """    

    cell = hoomd.md.nlist.Cell(buffer=buffer_size)
    cutoff = (1.0 + 20*debye_length) * (debye_length==0 or energy_scale==0)
    dlvo = hoomd.md.pair.DLVO(nlist=cell, default_r_cut=cutoff)
    dlvo.params[('A', 'A')] = dict(A = 0, a1 = 0.5, a2 = 0.5, kappa = 1.0/debye_length, Z = 4*energy_scale)
    
    return dlvo


def capped_dlvo(debye_length:float, energy_scale:float, buffer_size:float=0.4, force_cap:float|None=None):
    """
    Creates a `hoomd.md.pair.Table <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/table.html>`_ object with the given debye length and energy scale. The DLVO interaction is a screened electrostatic potential for simulating charged colloids in an electrolyte, but sometimes generates impractically large forces and thus displacements. Therefore, this method creates a tabular potential where high forces are clipped to keep simulations running smoothely. However, using this method may result in occasional overlapping particles.
    
    :param debye_length: The debye length in simulation units
    :type debye_length: float
    :param energy_scale: The prefactor on the exponential energy scale of the interaction in kT units
    :type energy_scale: float
    :param buffer_size: The buffer size for the neighbor list, defaults to 0.4
    :type buffer_size: float, optional
    :param force_cap: the maximum particle-particle force particles may experience in a simulation, defaults to a force which results in only displacements as big as fifteen debye lengths (given the diffusivity is 0.25 and the timestep is 1e-3).
    :type force_cap: float | None, optional
    :return: A hoomd `md.pair.Table` object
    :rtype: hoomd.md.pair.Pair
    """    

    cell = hoomd.md.nlist.Cell(buffer=buffer_size)
    nonideal = debye_length!=0 and energy_scale!=0
    cutoff = (1.0 + 20*debye_length) * nonideal

    r = np.linspace(0,cutoff,10000, endpoint=False)
    dr = r[1]-r[0]

    if force_cap is None:
        force_cap = 1.0*(15*debye_length) / (0.25*1e-3)

    forces = np.clip( energy_scale/debye_length * np.exp(-(r-1.0)/debye_length),-force_cap,force_cap)

    energies = np.flip(np.cumsum(np.flip(forces)*dr))
    dlvo = hoomd.md.pair.Table(nlist=cell,default_r_cut=cutoff)
    dlvo.params[('A','A')] = {'r_min':0.0,'U':energies-energies.min(),'F':forces}

    return dlvo


def hoomd_wca(length_scale:float, energy_scale:float, buffer_size:float=0.4):
    """Creates a `hoomd.md.pair.LJ <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/lj.html>`_ object with the given length scale and energy scale. This Lennard-Jones interaction is trucated to the Weeks-Chandler-Anderson form for purely repuslive particles. This interaction has the form:

    .. math::
        U(r) = 4\\varepsilon\\bigg[(\\sigma/r)^{{12}} - (\\sigma/r)^6 + 1\\bigg]

    :param length_scale: the length scale of the WCA interaction in simulation units
    :type length_scale: float
    :param energy_scale: the energy scale of the WCA interaction in kT units
    :type energy_scale: float
    :param buffer_size: the buffer size for the neighbor list, defaults to 0.4
    :type buffer_size: float, optional
    :return: a `hoomd.md.pair.LJ <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/lj.html>`_ object with the given length scale and energy scale
    :rtype: hoomd.md.pair.LJ
    """    
    
    nonideal = length_scale!=0 and energy_scale!=0
    cutoff = (2**(1/6) * length_scale)*nonideal
    cell = hoomd.md.nlist.Cell(buffer=buffer_size)
    wca = hoomd.md.pair.LJ(nlist=cell, default_r_cut=cutoff,mode='shift')
    wca.params[('A', 'A')] = dict(epsilon=energy_scale,sigma=length_scale)

    return wca


def hoomd_alj(shape:SuperEllipse, energy_scale:float, buffer_size:float=0.4, **kwargs):
    """Creates a `hoomd.md.pair.aniso.ALJ <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/aniso.html#hoomd.md.pair.aniso.ALJ>`_ object with the given shape and energy scale. The Anisotropic Lennard-Jones interaction is a generalization of the Lennard-Jones interaction to arbitrary shapes. This interaction has the form:

    .. math::
        U(r) = 4\\varepsilon\\bigg[(\\sigma_i\\sigma_j/r^2)^{{6}} - (\\sigma_i\\sigma_j/r^2)^3 + 1\\bigg]

    :param shape: the shape of particles to generate an ALJ interaction for, defaults to a disc with diameter 1.0
    :type shape: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>`
    :param energy_scale: the energy scale of the ALJ interaction in kT units
    :type energy_scale: float
    :param buffer_size: the buffer size for the neighbor list, defaults to 0.4
    :type buffer_size: float, optional
    :return: a `hoomd.md.pair.aniso.ALJ <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/aniso.html#hoomd.md.pair.aniso.ALJ>`_ object with the given shape and energy scale
    :rtype: hoomd.md.pair.aniso.ALJ
    """    

    if not hasattr(shape, 'vertices'):
        assert ('contact_radius' in kwargs) and ('n_verts' in kwargs), "must include arguments to make shape vertice if not pre-generated"
        shape.contact_vertices(n_verts=kwargs['n_verts'],contact_ratio=kwargs['contact_radius']/shape.ay)
    
    if 'contact_radius' in kwargs:
        alj_contact = kwargs['contact_radius']
    else:
        alj_contact = shape.contact_ratio*shape.ay
    
    sigma_core = shape.core_radius
    sigma_out = shape.outsphere

    nonideal = sigma_core!=0 and energy_scale!=0
    try:
        cutoff = 2.0**(1.0/6.0)*sigma_core*(1.1*sigma_out/sigma_core)*nonideal
    except ZeroDivisionError:
        cutoff = 0

    cell = hoomd.md.nlist.Cell(buffer=buffer_size)
    alj = hoomd.md.pair.aniso.ALJ(cell, default_r_cut=cutoff)

    vertices = shape.vertices.tolist()
    faces = [[i for i,_ in enumerate(vertices)]]
    alj.shape['A'] = dict(vertices = vertices, faces=faces, rounding_radii=[0.0,0.0,0.0])

    alj.params[('A','A')] = dict(
        epsilon = energy_scale,
        sigma_i = sigma_core,
        sigma_j = sigma_core,
        contact_ratio_i = alj_contact,
        contact_ratio_j = alj_contact,
        alpha = 0)

    return alj

def hpmc_dipoles(shape:SuperEllipse, energy_scale:float):
    """
    Creates a pair of dipole interactions for the given shape and energy scale.

    :param shape: the shape of particles to generate dipole fields for, defaults to a disc with diameter 1.0
    :type shape: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>`
    :param energy_scale: the energy scale of the dipole interactions in kT units
    :type energy_scale: float
    :return: a tuple of `hoomd.hpmc.pair.AngularStep <https://hoomd-blue.readthedocs.io/en/latest/hoomd/hpmc/pair/angularstep.html>`_ objects which represent the dipole interactions
    :rtype: tuple[hoomd.hpmc.pair.AngularStep, hoomd.hpmc.pair.AngularStep]
    """

    ax, ay = shape.ax, shape.ay
    edge_width = 1.0
    r_frac = 2.5

    r_a = r_frac*np.sqrt(ax**2 + (ay/edge_width)**2)
    d_a = np.arctan(ay/ax/edge_width)

    attract = hoomd.hpmc.pair.AngularStep(hoomd.hpmc.pair.Step())
    attract.isotropic_potential.params[('A','A')] = dict(epsilon=[-energy_scale], r = [r_a])
    attract.mask[('A','A')] = dict(directors = [(1.0,0.0,0.0), (-1.0,0.0,0.0)], deltas=[d_a]*2)

    r_r = r_frac*np.sqrt(ay**2 + (ax/edge_width)**2)
    d_r = np.arctan(ax/ay/edge_width)

    repel = hoomd.hpmc.pair.AngularStep(hoomd.hpmc.pair.Step())
    repel.isotropic_potential.params[('A','A')] = dict(epsilon=[energy_scale], r = [r_r])
    repel.mask[('A','A')] = dict(directors = [(0.0,1.0,0.0), (0.0,-1.0,0.0)], deltas=[d_r]*2)

    return attract, repel

# class TypeUpdater(hoomd.custom.Action):
#     def __init__(self, bounds:float|list|np.ndarray,direct:list|np.ndarray=np.array([0,np.pi/2])):
#         self._bdry = None
#         self.bounds = bounds
#         self._u = np.array([np.cos(direct),np.sin(direct),np.zeros_like(direct)]).T

#     @property
#     def bounds(self) -> np.ndarray:
#         return self._bdry
    
#     @bounds.setter
#     def bounds(self, bounds:float|list|np.ndarray):
#         if isinstance(bounds, float):
#             self._bdry = np.array([bounds])
#         elif isinstance(bounds, np.ndarray):
#             self._bdry = bounds
#         elif isinstance(bounds, list):
#             self._bdry = np.array(bounds)

#     @property
#     def unit_vectors(self) -> np.ndarray:
#         return self._uvec
    
#     @unit_vectors.setter
#     def unit_vectors(self, direct:list|np.ndarray):
#         self._u = np.array([np.cos(direct),np.sin(direct),np.zeros_like(direct)]).T

#     def attach(self, simulation):
#         self._state = simulation.state
#         self._comm = simulation.device.communicator

#     # def detach(self):
#     #     del self._state
#     #     del self._comm

#     def act(self, timestep):
#         with self._state.cpu_local_snapshot as snap:
#             pts = snap.particles.position
#             proj = np.array([pts@u for u in self._u])
#             radii = np.linalg.norm(proj, axis=0)
#             snap.particles.typeid = np.digitize(radii,self._bdry,right=True)
