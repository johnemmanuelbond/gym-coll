# -*- coding: utf-8 -*-
"""
Contains a few methods to simplify interacting with `hoomd-blue <https://hoomd-blue.readthedocs.io/en/latest>`_. Includes a class to represent electrode geometries.
"""
import numpy as np
from scipy.spatial.distance import squareform, pdist

import gsd.hoomd

import importlib.util, warnings
has_hoomd = False
try:
    spec = importlib.util.find_spec('hoomd')
    if spec is not None:
        has_hoomd=True
except ModuleNotFoundError:
    has_hoomd = False
    warnings.warn("hoomd not found, sims.bd module will not work. Install hoomd-blue to use this module.")
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

def electrode_logger(k_trans:float|list|np.ndarray,
                     k_rot:float|list|np.ndarray,
                     direct:float|list|np.ndarray,
                     electrode_gap:float,):
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
    k_trans = np.array([k_trans]).flatten().tolist()
    k_rot   = np.array([k_rot]).flatten().tolist()
    direct  = np.array([direct]).flatten().tolist()

    action_log = hoomd.logging.Logger(only_default=False)
    action_log[('electrode','k_trans')] = (lambda: k_trans, 'sequence')
    action_log[('electrode','k_rot')] = (lambda: k_rot, 'sequence')
    action_log[('electrode','direct')] = (lambda: direct, 'sequence')
    action_log[('electrode','dg')] = (lambda: electrode_gap, 'scalar')
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
    dlvo.params[('Z','Z')] = {'r_min':0.0,'U':energies-energies.min(),'F':forces}

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
    wca.params[('Z', 'Z')] = dict(epsilon=energy_scale,sigma=length_scale)

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
    
    sigma_core = 2*shape.core_radius
    sigma_out = shape.outsphere
    alj_contact = shape.contact_ratio*shape.ay*2/sigma_core

    nonideal = sigma_core!=0 and energy_scale!=0
    try:
        cutoff = 2.0**(1.0/6.0)*sigma_core*(1.1*sigma_out/sigma_core)*nonideal
    except ZeroDivisionError:
        cutoff = 0

    # lmin = 2.0 ** (1.0/6.0)
    # suggested_cutoff = max(lmin*sigma_core, outsphere + lmin/2(β i⋅σi +βj ⋅σj))
    # print(cutoff, suggested_cutoff)


    cell = hoomd.md.nlist.Cell(buffer=buffer_size)
    alj = hoomd.md.pair.aniso.ALJ(cell, default_r_cut=cutoff)

    vertices = shape.vertices.tolist()
    faces = [[i for i,_ in enumerate(vertices)]]
    alj.shape['Z'] = dict(vertices = vertices, faces=faces, rounding_radii=[0.0,0.0,0.0])
    alj.params[('Z','Z')] = dict(
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
    attract.isotropic_potential.params[('Z','Z')] = dict(epsilon=[-energy_scale], r = [r_a])
    attract.mask[('Z','Z')] = dict(directors = [(1.0,0.0,0.0), (-1.0,0.0,0.0)], deltas=[d_a]*2)

    r_r = r_frac*np.sqrt(ay**2 + (ax/edge_width)**2)
    d_r = np.arctan(ax/ay/edge_width)

    repel = hoomd.hpmc.pair.AngularStep(hoomd.hpmc.pair.Step())
    repel.isotropic_potential.params[('Z','Z')] = dict(epsilon=[energy_scale], r = [r_r])
    repel.mask[('Z','Z')] = dict(directors = [(0.0,1.0,0.0), (0.0,-1.0,0.0)], deltas=[d_r]*2)

    return attract, repel


class SwitchEta(hoomd.custom.Action):
    def __init__(self, eta_bins:np.ndarray, shape:SuperEllipse, periodic=[False, False, False]):
        self.shape = shape
        self._bdry = eta_bins

    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator
        self._Nt = len(self._state.particle_types)

    @property
    def shape(self):
        return self._s

    @shape.setter
    def shape(self, shape:SuperEllipse):
        if not hasattr(shape, 'outsphere'): shape.contact_vertices(n_verts=16, require_corners=True)
        self._s = shape
        self._d = shape.outsphere
        self._Ap = shape.area
    
    def local_eta(self, pts, box):
        dists = squareform(pdist(pts))
        ncut = 2.6*self._d
        nnei_inner = np.sum(dists<(ncut-self._d/2), axis=-1)
        nnei_outer = np.sum(dists<(ncut+self._d/2), axis=-1)
        nnei = nnei_inner + 0.5*(nnei_outer - nnei_inner)
        etas = nnei * self._Ap / (np.pi*ncut**2)
        return etas

    def act(self, timestep):
            
        with self._state.cpu_local_snapshot as snap:
            pts = snap.particles.position
            box = snap.local_box.L
            etas = self.local_eta(pts, box)
            type_idx = np.digitize(etas, self._bdry).clip(1, self._Nt) - 1
            snap.particles.typeid = type_idx

