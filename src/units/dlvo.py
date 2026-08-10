# -*- coding: utf-8 -*-
"""
Contains a few helper methods for converting SI values :math:`[m,s,J]` to simulation values :math:`[l,\\tau,kT]`. Most of these functions take in a series of keyword arguments which are best organized into a dictionary, such as :code:`kwargs = units.DEFAULT_PHYSICAL_PARAMETERS`, which can be passed into functions with the unpacking operator, :code:`func(**kwargs)`. The functions below have sensible default values, but kwargs needs to contain the specific keys indicated by the function documentation in order to overwrite the defaults. Importantly, all input arguments *must* have SI units (no prefixes!).
"""

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

# General physical constants
kb = 1.380e-23 # [J/K] Boltzmann constant
e = 1.602177e-19 # [C] elementary charge
eps = 8.854e-12 # [F/m] or [J/(m*V**2)], permittivity of free space
Na = 6.0221408e23 # [1/mol] Avogadro's number


DEFAULT_PHYSICAL_PARAMETERS = {
    "temperature": 298,
    "rel_perm_m": 78,
    "ion_multiplicity": 1,
    "debye_length": 30.0e-09,
    "viscosity": 0.0008931,
    "particle_radius": 1.435e-06,
    "particle_density": 1980,
    "surface_potential": -50.0e-03,
    "fcm": -0.4667,
    "voltage": 0.0,
    "electrode_gap": 91.0e-06,
    "fps": 8,
    }

# WIP, POTENTIALLY BEYOND THE SCOPE OF THIS MODULE
# def VdW_prefactor(particle_radius=1e-6, hamaker_constant=0.0, vdw_power=1.0, temperature = 293, **kwargs):
#     kT = kb*temperature
#     a = particle_radius
#     return 1/12 * hamaker_constant * a * (2*a)**(-vdw_power) / kT


def rel_gravity(particle_volume=None,particle_density=1960,solution_density=1000,g_acc=9.807,temperature=293,**kwargs):
    """
    The force due to gravity, in simulation units, of a colloidal particle suspended in a medium. Depends on the acceleration due to gravity, the particle and medium densities, and the particle volume. Also depends on the temperature and particle radius for unit conversions.

    :param particle_volume: volume of the particle in SI units, defaults to a 1-micron sphere
    :type particle_volume: scalar, optional
    :param particle_density: density of the particle in SI units, defaults to 1960 for silica
    :type particle_density: scalar, optional
    :param solution_density: density of the medium in SI units, defaults to 1000 for water
    :type solution_density: scalar, optional
    :param g_acc: acceleration due to gravity, defaults to 9.807 for earth at sea level
    :type g_acc: float, optional
    :param temperature: the temperature in K, defaults to 293
    :type temperature: scalar, optional
    :return: the force due to gravity (including buoancy) in simulation units [kT/2a]
    :rtype: scalar
    """
    if particle_volume is None:
        if 'particle_radius_x' in kwargs and 'particle_radius_y' in kwargs:
            particle_volume = np.pi*kwargs['particle_radius_x']*kwargs['particle_radius_y']*kwargs['particle_radius_z']*2
        elif 'particle_radius' in kwargs:
            particle_volume = 4/3 * np.pi * kwargs['particle_radius']**3
        else:
            particle_volume = 4/3 * np.pi * (0.5e-6)**3

    G_rel = particle_volume * (particle_density - solution_density) * g_acc
    
    if 'particle_radius_y' in kwargs:
        SI_2a  = 2*kwargs['particle_radius_y'] # simulation length unit
    else:
        SI_2a  = 2*kwargs['particle_radius'] # simulation length unit
    SI_kT  = kb*temperature # simulation energy unit

    return -G_rel * (SI_2a/SI_kT)


def dlvo_prefactor(particle_radius=1.0e-6, surface_potential=-50e-3,temperature=298,ion_multiplicity=1,rel_perm_m=78,**kwargs):
    """
    Calculates the prefactor on a screned electrostatic repulsion between spherical particles in kT based on given experimental conditions. Depends on the particle size and surface potential, the temperature, and the permittivity and ion multiplicity of the screening solution.

    :param particle_radius: the radius of the interacting colloidal spheres in [m], defaults to 1 micron
    :type particle_radius: scalar, optional
    :param surface_potential: the electric potential at the surface of the colloidal spheres in [V], defaults to -50mV
    :type surface_potential: scalar, optional
    :param temperature: the absolute temperture in [K], defaults to 298K
    :type temperature: scalar, optional
    :param ion_multiplicity: the (unitless) ion multiplicity of the screening solution, defaults to 1
    :type ion_multiplicity: int, optional
    :param rel_perm_m: the (unitless) permittivity of the medium, defaults to 78 for water
    :type rel_perm_m: scalar, optional
    :return: the prefactor on a screned electrostatic repulsion between spheres in [kT] units
    :rtype: scalar
    """   

    a = particle_radius         # [m]
    psi = surface_potential     # [V]
    kT = kb*temperature         # [J]
    ze = ion_multiplicity*e     # [C]
    rel_eps = rel_perm_m*eps    # [F/m] permittivity of sol'n

    #below gives bpp in joules [J]
    bpp = 32 * np.pi * rel_eps * a * ((kT/ze)**2) * np.tanh((ze*psi) / (4*kT))**2

    return bpp/kT  # converts from [J] to units of [kT]


def kappa(particle_radius=1.0e-6, debye_length=30e-9,**kwargs):
    """
    Calculates the (unitless) decay constant for screened electrostatic repulsion between spherical colloids based on experimental conditions. Unless an explicit debye length is provided, this value depends on the \'temperature\', and the \'permittivity\', \'ion multiplicity\', and \'electrolyte concentration\' of the solution found in :code:`**kwargs`.

    :param particle_radius: the radius, in [m] of the interacting colloidal spheres, defaults to 1 micron
    :type particle_radius: scalar, optional
    :param debye_length: the electrostatic screening length in [m], defaults to 30 nm for silica spheres in room-temp water.
    :type debye_length: scalar, optional
    :return: the unitless decay constant used in screened electrostatic interaction potentials: :math:`[2a/\\lambda_D]`
    :rtype: scalar
    """    
    
    if debye_length is None:
        has_t = ('temperature' in kwargs)
        has_im = ('ion_multiplicity' in kwargs)
        has_rp = ('rel_perm_m' in kwargs)
        has_ec = ('electrolyte_concentration' in kwargs)
        assert has_t and has_ec and has_rp and has_im, "without a given debye length, please input a \'temperature\', \'ion_mulitplicity\', \'rel_perm_m\', and \'electrolyte_concentration\'"

        kT = kb*kwargs['temperature']  # [J]
        ze = kwargs['ion_multiplicity']*e        # [C] electrolyte charge
        rel_eps = kwargs['rel_perm_m']*eps # [F/m] permittivity of sol'n
        C = kwargs['electrolyte_concentration']  # [mol/L]

        #assuming a symmetric electrolyte
        dL = ((rel_eps*kT)/(2*(ze**2)*(C*1000*Na)))**(1/2)
        kap = 2*particle_radius/dL  #[1/2a]
    else:
        #assuming a symmetric electrolyte
        kap = 2*particle_radius/debye_length  #[1/2a]

    return kap


def dlvo_minimum(gravity_force=None, dlvo_pf=None, debye_length=None, **kwargs):
    """
    Returns the minimum-energy position of a particle levitating over a wall under the influence of DLVO electrotatics and gravity:

    .. math::

        h_m = \\lambda_D\\log(B/\\lambda_DF_g)
    
    Where :math:`B` is the energy scale associated with screened electrostatic repulsion between a particle and the a wall. Unless an explict dlvo energy scale is provided, this value depends on the \'particle radius\' and \'surface potential\', the \'temperature\', and the \'permittivity\' and \'ion multiplicity\' of the screening solution found in :code:`**kwargs`.
        
    :math:`\\lambda_D` is the debye length of the medium. Unless an explicit debye length is provided, this value depends on the \'temperature\', and the \'permittivity\', \'ion multiplicity\', and \'electrolyte concentration\' of the solution found in :code:`**kwargs`.

    :math:`F_g` is the force due to gravity of the particle in the medium. Unless an explicit force is provided, this quantity depends on the acceleration due to gravity, the particle and medium densities, and the particle volume found in :code:`**kwargs`.

    :param gravity_force: force due to gravity in simulation units [kT/2a], defaults to a 1-micron silica particle in water.
    :type gravity_force: scalar, optional
    :param dlvo_pf: energy scale of dlvo electrostatics, defaults to a silica colloid in water
    :type dlvo_pf: scalar, optional
    :param debye_length: the debye length in either SI or simulation units [2a], defaults to 30nm
    :type debye_length: scalar, optional
    :return: the minimum-energy position of a colloid above the wall in simulation units [2a]
    :rtype: scalar
    """    
    if gravity_force is None:
        gravity_force = rel_gravity(**kwargs)
    if dlvo_pf is None:
        dlvo_pf = 2*dlvo_prefactor(**kwargs)
    
    if debye_length is None:
        kap = kappa(**kwargs)
    elif np.log(debye_length) > 0:
        kap = 1/debye_length
    else:
        kap = kappa(debye_length=debye_length,**kwargs)

    return np.log(-2*kap*dlvo_pf/gravity_force)/kap

