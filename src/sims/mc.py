# -*- coding: utf-8 -*-
"""Monte Carlo simulations for hard colloidal particles in 2D.

This module uses HOOMD's HPMC integrators to simulate hard-particle systems
with Monte Carlo moves. In the usual Metropolis scheme, a trial move from an
old state :math:`x` to a new state :math:`x'` is accepted with probability

.. math::

    P_\\mathrm{acc}(x \\rightarrow x') = \\min\\left(1, e^{-\\Delta U / kT}\\right),

where :math:`\\Delta U` is the change in potential energy between states :math:`x` and 
:math:`x'`. For hard particles, this reduces to rejecting any move that causes overlap 
and accepting moves that preserve the excluded-volume constraint.

Because the environments in this project are typically evolved forward in time,
this implementation restricts the number of Monte Carlo trial moves to mimic a
Brownian dynamics-like stepping scheme. The step sizes are therefore tied to
internal diffusivities in the same spirit as :py:class:`BrownianDynamics <sims.bd.BrownianDynamics>`.

HOOMD's HPMC integrators sample translation moves from within a hypersphere of
radius :math:`dx` and rotation moves uniformly from the interval
:math:`[-da, da]`. The values of :math:`dx` and :math:`da` are updated from
internal translational and rotational diffusivities, :math:`D_T` and :math:`D_R`.

The simulation uses the same internal unit conventions as the Brownian
Dynamics module: energy is expressed in units of :math:`kT`, length is set by
the particle geometry, and time is controlled by the diffusivity scale.
"""
import numpy as np
import hoomd
from sims import HoomdColloid
from utils import SuperEllipse

_default_sphere = SuperEllipse(ax=0.5,ay=0.5)

class DynamicMonteCarlo(HoomdColloid):
    """A generic class for 2D Monte Carlo simulations.

    :param dt: timestep for the Monte Carlo integrator, defaults to 1e-3
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
                 types:int = 1,
                 shape:SuperEllipse = _default_sphere):
        """Initialize the Monte Carlo simulation."""
        super().__init__(dt=dt, DT=DT, DR=DR, kT=kT, num_types=types, shape=shape)
        self._ideal = False

        self._pair = []
        self._external = []
        self._hpmc = self._shape_integrator()
        self._alt_methods = False
    
    @property
    def dx(self) -> float:
        """
        :return: the maximum translational step size in simulation units
        :rtype: scalar
        """
        return np.sqrt(4*self.DT*self.dt)

    @dx.setter
    def dx(self, dx:float):
        """
        :param dx: the maximum translational step size in simulation units
        :type dx: scalar
        """
        self._DT = dx**2/(4*self.dt)

    @property
    def da(self) -> float:
        """
        :return: the maximum rotational step size in simulation units
        :rtype: scalar
        """
        return np.sqrt(6*self.DR*self.dt)

    @da.setter
    def da(self, da:float):
        """
        :param da: the maximum rotational step size in simulation units
        :type da: scalar
        """
        self._DR = da**2/(6*self.dt)

    def _shape_integrator(self) -> hoomd.hpmc.integrate.HPMCIntegrator:
        """Return a default HPMC integrator object with the current step sizes."""
        if self._is_disc:
            mc = hoomd.hpmc.integrate.Sphere(nselect=2, translation_move_probability=1.0,default_d=self.dx)
        else:
            mc = hoomd.hpmc.integrate.ConvexSpheropolygon(nselect=3,translation_move_probability=2/3,default_a=self.da,default_d=self.dx)
        return mc

    @property
    def integrator(self) -> hoomd.hpmc.integrate.HPMCIntegrator:
        """
        :return: the current Monte Carlo integrator object
        :rtype: :py:class:`hoomd.hpmc.integrate.HPMCIntegrator`
        """
        mc = self._hpmc

        if self._ideal:
            for p1,p2 in self.pairs:
                mc.interaction_matrix[(p1,p2)] = False

        if set(mc.shape.keys()) != set(self.types):
            for t in self.types:
                if self._is_disc:
                    mc.shape[t] = dict(diameter=2*self._s.ay)
                else:
                    mc.shape[t] = dict(vertices = self._s.vertices[:,:2].tolist(), sweep_radius = self._s.contact_ratio*self._s.ay)

        if len(self._pair) > 0: mc.pair_potentials = self._pair
        if len(self._external) > 0: mc.external_potentials = self._external

        return mc

    @integrator.setter
    def integrator(self, hpmc: hoomd.hpmc.integrate.HPMCIntegrator|None):
        """
        :param value: the Monte Carlo integrator to use
        :type value: :py:class:`hoomd.hpmc.integrate.HPMCIntegrator`
        """
        if hpmc is None:
            self._hpmc = self._shape_integrator()
        else:
            assert isinstance(hpmc, hoomd.hpmc.integrate.Integrator), "integrator must be a hoomd.hpmc.integrate.Integrator object"
            self._hpmc = hpmc
        

    @property
    def ideal(self) -> bool:
        """
        :return: whether the simulation ignores pair interactions
        :rtype: bool
        """
        return self._ideal
    
    @ideal.setter
    def ideal(self, value: bool):
        """
        :param value: whether the simulation ignores pair interactions
        :type value: bool
        """
        self._ideal = value
    
    @property
    def externals(self) -> list:
        """
        :return: the list of external potentials applied to the simulation
        :rtype: list of :py:class:`hoomd.hpmc.external.External` objects
        """
        return self._external

    @externals.setter
    def externals(self, value: list):
        """
        :param value: the list of external potentials to apply to the simulation
        :type value: list of :py:class:`hoomd.hpmc.external.External` objects
        """
        assert isinstance(value,list) and all(isinstance(x, hoomd.hpmc.external.External) for x in value), "externals must be a list of potential objects"
        self._external = value

    @property
    def pair_potentials(self) -> list:
        """
        :return: the list of particle-particle pair potentials applied to the simulation
        :rtype: list of :py:class:`hoomd.hpmc.pair.Pair` objects
        """
        return self._pair

    @pair_potentials.setter
    def pair_potentials(self, value: list):
        """
        :param value: the list of particle-particle pair potentials to apply to the simulation
        :type value: list of :py:class:`hoomd.hpmc.pair.Pair` objects
        """
        assert isinstance(value,list) and all(isinstance(x, hoomd.hpmc.pair.Pair) for x in value), "pairs must be a list of pair potential objects"
        self._pair = value

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
            if not self._is_disc: self.logger.add(self._hpmc,quantities=['type_shapes'])

        super().run(time, *args)

if __name__ == "__main__":
    pass