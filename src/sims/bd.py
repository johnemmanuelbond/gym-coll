# -*- coding: utf-8 -*-
"""Brownian dynamics simulations for colloidal particles in 2D.

This module implements Brownian dynamics with HOOMD-blue for quasi-2D
colloidal systems. The dynamics are driven by thermal noise and a particle-
particle interaction potential, and the state of the system is represented by
particle positions and, for anisotropic shapes, orientations.

In brownian dynamics, the time evolution the each coordinate, :math:`x`, is 
given by the overdamped Langevin equation:

.. math::

    x(t+dt) = x(t) + D\\partial_x U/kT dt + \\Gamma\\sqrt{2Ddt}

where :math:`x` may be the position or orientation of any particle and :math:`D` 
is the diffusivity along that coordinate (translation or rotation). :math`\\partial_x U/kT` 
is the net force acting along that coordinate compared to :math:`kT`, the thermal energy. 
The force comes from a derivative of the total potential energy, :math:`U`, which may 
include pairwise interactions and external potentials. :math:`\\Gamma` is a random 
variable representing Brownian motion.

The simulation uses a set of internal units for length, time, and energy. For example, 
it is common to set simulation units to 1.0 to avoid memory errors. In BD, :math:`kT=1.0` 
is the natural energy unit and the particle diameter, :math:`2a=1.0`, is the natural 
length unit. The time unit is controlled by the translational diffusivity. 
Commonly, setting math:`D_T=0.25` automatically sets the diffusive time scale 
to :math:`\\tau = a^2/D_T = 1.0` which is convenient.

However, it is also helpful to express simuations in physical units for comparison 
against experiments. Particle sizes, :math:`a` and diffusivities :math:`D_T` can be 
simply, if not easily, measured. And so expressing the particle dimensions in microns 
and the diffusivity in :math:`\\mu m^2/s` automatically sets the time unit to be seconds.

For both these cases, the :py:mod:`pchem.units <pchem.units>` module provides a 
convenient interface to convert between simulation units and physical units.

The interaction between particles is described by a HOOMD pair potential,
and additional forces can be attached through the ``forces`` interface.
"""

import numpy as np

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

from sims import HoomdColloid
from utils import SuperEllipse, hoomd_wca, hoomd_alj

_default_sphere = SuperEllipse(ax=0.5,ay=0.5)

class BrownianDynamics(HoomdColloid):
    """A generic class for 2D Brownian dynamics simulations.

    :param dt: timestep for the Brownian integrator, defaults to 1e-3
    :type dt: scalar, optional
    :param DT: short-time translational diffusivity in simulation units
    :type DT: scalar, optional
    :param DR: short-time rotational diffusivity in simulation units
    :type DR: scalar, optional
    :param kT: temperature in energy units, defaults to 1.0
    :type kT: scalar, optional
    :param types: number of particle types, defaults to 1
    :type types: int, optional
    :param shape: particle shape, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <utils.geometry.SuperEllipse>` object, optional
    """

    def __init__(self,
                 dt:float = 1e-3,
                 DT:float = 0.25,
                 DR:float = 0.025,
                 kT:float = 1.0,
                 num_types = 1,
                 shape:SuperEllipse = _default_sphere):
        """Initialize the Brownian dynamics simulation."""
        super().__init__(dt=dt, DT=DT, DR=DR, kT=kT, num_types=num_types, shape=shape)

        if self._is_disc:

            self._Uij = hoomd_wca(1.0,0.0)
        else:
            self._Uij = hoomd_alj(self._s,0.0)

        self._forces = []
        self._methods = [self._default_BD()]
    
    def _default_BD(self) -> hoomd.md.methods.Brownian:
        """Return a default Brownian dynamics method object with the current diffusivities and temperature."""
        return hoomd.md.methods.Brownian(filter=hoomd.filter.All(),kT=self._kT,
                                         default_gamma   =  self._kT/self._DT,
                                         default_gamma_r = [self._kT/self._DR]*3)

    @property
    def integrator(self) -> hoomd.md.Integrator:
        """In BD, the integrator object contains methods like BD and forces like pair potentials. Therefore it automatically applies the current :py:attr:`interaction` and :py:attr:`forces` to the simulation, and is thus not a settable property. 

        :return: The current Brownian dynamics integrator object
        :rtype: `hoomd.md.Integrator <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/integrate/integrator.html>`_
        """
        if not self._is_disc:
            for t in self.types:
                self._Uij.shape[t] = self._Uij.shape['Z']

        for p1,p2 in self.pairs:
            self._Uij.params[(p1,p2)] = self._Uij.params[('Z','Z')]

        return hoomd.md.Integrator(dt=self._dt, methods = self._methods, forces=[self._Uij,*self._forces], integrate_rotational_dof = not self._is_disc)

    @property
    def methods(self):
        """
        :return: the list of current integration methods objects
        :rtype: list of `hoomd.md.methods.Method <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/methods/method.html>`_ objects """
        return self._methods

    @methods.setter
    def methods(self, methods:list|None):
        """ :param methods: the integration method or methods to use :type methods: `hoomd.md.methods.Method <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/methods/method.html>`_ or a list thereof """
        if methods is None or len(methods)==0:
            self._methods = [self._default_BD()]

        assert isinstance(methods,list) and all([isinstance(m,hoomd.md.methods.Method) for m in methods]), "methods must be a list of hoomd.md.methods.Method objects"
        self._methods = methods

    @property
    def core(self):
        """ :return: the core particle-particle interaction object :rtype: `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/pair.html>`_ """
        return self._Uij

    @core.setter
    def core(self, pair_potential):
        """ :param pair_potential: the core particle-particle interaction to use :type pair_potential: `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/pair/pair.html>`_ """
        assert isinstance(pair_potential, hoomd.md.pair.Pair), "interaction must be a hoomd.md.pair.Pair object"
        self._Uij = pair_potential

    @property
    def ideal(self) -> bool:
        """ :return: whether the interaction range is zero for all particle pairs :rtype: bool """
        rcuts = np.array([self._Uij.r_cut[p1,p2] for p1,p2 in self.pairs])
        return np.all(rcuts==0.0)

    @ideal.setter
    def ideal(self, value: bool):
        """ :param value: whether the simulation should use zero interaction range for all pairs :type value: bool """
        if not value: raise Exception("cannot set ideal to False, reset the interaction to a nonzero potential instead")
        for p1,p2 in self.pairs:
            self._Uij.r_cut[p1,p2] = 0.0
    
    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, forces:list):
        """ :param forces: the extra forces to apply during integration :type forces: list of `hoomd.md.force.Force <https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/force/force.html>`_ objects """
        assert isinstance(forces,list) and all([isinstance(f, hoomd.md.force.Force) for f in forces]), "forces must be a list of hoomd.md.force.Force objects"
        self._forces = forces

    def run(self,
            time:float,
            *args):
        """Advance the simulation for a short burst.

        :param time: runtime in simulation units for this burst
        :type time: scalar
        :param args: additional arguments passed to the simulation
        :type args: tuple
        """

        if len(self._sim.operations.writers)>0:
            if not self._is_disc: self.logger.add(self._Uij,quantities=['type_shapes'])

        super().run(time, *args)


if __name__ == "__main__":
    pass