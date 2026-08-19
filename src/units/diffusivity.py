# -*- coding: utf-8 -*-
"""
Contains a few helper methods for converting SI values :math:`[m,s,J]` to simulation values :math:`[l,\\tau,kT]`. Most of these functions take in a series of keyword arguments which are best organized into a dictionary, such as :code:`kwargs = units.DEFAULT_PHYSICAL_PARAMETERS`, which can be passed into functions with the unpacking operator, :code:`func(**kwargs)`. The functions below have sensible default values, but kwargs needs to contain the specific keys indicated by the function documentation in order to overwrite the defaults. Importantly, all input arguments *must* have SI units (no prefixes!).
"""

import numpy as np
# from utils import SuperEllipse

# General physical constants
kb = 1.380e-23 # [J/K] Boltzmann constant

def sphere_trans_diff(temperature=298, viscosity=0.8931e-3, particle_radius=1.0e-6,**kwargs):
    """
    Computes the translational diffusivity (in SI units) for a sphere given experimental conditions

    :param temperature: the absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param viscosity: the viscosity of the medium [Pa s], defaults to 0.8931 mPa s
    :type viscosity: scalar, optional
    :param particle_radius: the radius of the spherical particle in [m], defaults to 1 micron
    :type particle_radius: scalar, optional
    :return: the translational diffusivity in [m^2/s]
    :rtype: scalar
    """
    return kb*temperature/(6*np.pi*viscosity*particle_radius)

def sphere_rot_diff(temperature=298, viscosity=0.8931e-3, particle_radius=1.0e-6,**kwargs):
    """
    Computes the rotational diffusivity (in SI units) for a sphere given experimental conditions

    :param temperature: the absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param viscosity: the viscosity of the medium [Pa s], defaults to 0.8931 mPa s
    :type viscosity: scalar, optional
    :param particle_radius: the radius of the spherical particle in [m], defaults to 1 micron
    :type particle_radius: scalar, optional
    :return: the rotational diffusivityin [rad^2/s]
    :rtype: scalar
    """
    return kb*temperature/(8*np.pi*viscosity*particle_radius**3)

def tau_sphere(sim_length=None, hydrodynamic_correction =1.0, temperature=298, viscosity=0.8931e-3, particle_radius=1.0e-6, **kwargs):
    """
    Computes the diffusive timescale (in SI units) for a sphere given experimental conditions. The timescale is computed as:

    .. math::

        t = \\frac{D_0^t\\sigma^2}{D^t}

    where :math:`D_0^t` is the simulated diffusivity (unitless, usually 0.25), :math:`D^t` is the translational diffusivity obtained via measurements or a model (SI), and :math:`\\sigma` is the characteristic length scale of the simulation (unitless, usually 1.0)

    Note that the translational diffusive timescale is also be used as the timescale for rotational dynamics of anisotropic particles.

    :param sim_length: the characteristic length scale of the simulation in [m], defaults to 2*particle_radius
    :type sim_length: scalar, optional
    :param hydrodynamic_correction: a (unitless) hydrodynamic correction to the single-particle diffusivity, defaults to 1.0 which has no effect
    :type hydrodynamic_correction: scalar, optional
    :param temperature: the absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param viscosity: the viscosity of the medium [Pa s], defaults to 0.8931 mPa s
    :type viscosity: scalar, optional
    :param particle_radius: the radius of the spherical particle in [m], defaults to 1 micron
    :type particle_radius: scalar, optional
    :return: the time it takes for the sphere to diffuse 1 length unit (sigma) in [s]
    :rtype: scalar
    """
    if sim_length is None:
        sim_length = 2*particle_radius

    expt_D0_t = hydrodynamic_correction * sphere_trans_diff(temperature=temperature, viscosity=viscosity, particle_radius=particle_radius)
    expt_tau_t = (sim_length**2)/(4*expt_D0_t)
    return expt_tau_t

def _scaled_ellipsoid(particle_radius_x=4e-6,particle_radius_y=2e-6, particle_radius_z=1e-6, **kwargs):
    az, ay, ax = np.sort([particle_radius_x, particle_radius_y, particle_radius_z]) 
    
    s = ax/az
    r = ax/ay
    assert s > 1 and r > 1, "Aspect ratios must be greater than 1"
    wp = ((r-1)*s)/((s-1)*r)

    return ax, s, wp

def shape_trans_diff(temperature=298, viscosity=0.8931e-3, particle_radius_x=4e-6,particle_radius_y=2e-6, particle_radius_z=1e-6, **kwargs):
    """
    Computes the translational diffusivity (in SI units) for an anisotropic particle given experimental conditions. Translational diffusion parallel and perpendicular to the principle particle axis is averaged into a single translational diffusion constant as :math:`D^t=(1/2)D_{\\parallel}^t+(1/2)D_{\\perp}^t`. Calculation is modeled after ellipsoidal particles according to WIP-QW.

    :param temperature: the absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param viscosity: the viscosity of the medium [Pa s], defaults to 0.8931 mPa s
    :type viscosity: scalar, optional
    :param particle_radius_x: the radius of the long axis of the ellipse in [m], defaults to 4 microns
    :type particle_radius_x: scalar, optional
    :param particle_radius_y: the radius of the short axis of the ellipse in [m], defaults to 2 micron
    :type particle_radius_y: scalar, optional
    :param particle_radius_z: the radius of the third axis of the ellipse in [m], defaults to 1 micron
    :type particle_radius_z: scalar, optional
    :return: the average translational diffusivity in [m^2/s]
    :rtype: scalar
    """
    ax, s, wp = _scaled_ellipsoid(particle_radius_x=particle_radius_x,particle_radius_y=particle_radius_y, particle_radius_z=particle_radius_z)

    sigP_t = np.log(s) + (s**2 + 15*s + 18)/(26*s + 8)
    sigO_t = np.log(s) + (-5*s**2 + 16*s + 30)/(33*s +8)
    sigma_t = wp * sigP_t + (1.0-wp) * sigO_t

    return sphere_trans_diff(temperature=temperature, viscosity=viscosity, particle_radius=ax) * sigma_t

def shape_rot_diff(temperature=298, viscosity=0.8931e-3, particle_radius_x=4e-6,particle_radius_y=2e-6, particle_radius_z=1e-6, **kwargs):
    """
    Computes the rotational diffusivity (in SI units) for an anisotropic particle about its smallest axis given experimental conditions.

    :param temperature: the absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param viscosity: the viscosity of the medium [Pa s], defaults to 0.8931 mPa s
    :type viscosity: scalar, optional
    :param particle_radius_x: the radius of the long axis of the ellipse in [m], defaults to 4 microns
    :type particle_radius_x: scalar, optional
    :param particle_radius_y: the radius of the short axis of the ellipse in [m], defaults to 2 micron
    :type particle_radius_y: scalar, optional
    :param particle_radius_z: the radius of the third axis of the ellipse in [m], defaults to 1 micron
    :type particle_radius_z: scalar, optional
    :return: the average translational diffusivity in [m^2/s]
    :rtype: scalar
    """
    ax, s, wp = _scaled_ellipsoid(particle_radius_x=particle_radius_x,particle_radius_y=particle_radius_y, particle_radius_z=particle_radius_z)

    sigP_r = np.log(s) + (2.0*s**2 + 13*s - 5.0)/(5*s + 5)
    sigO_r = np.log(s) + (-2.0*s**2 + 11*s + 12)/(15*s + 6)
    sigma_r = wp * sigP_r + (1-wp) * sigO_r

    return  sphere_rot_diff(temperature=temperature, viscosity=viscosity, particle_radius=ax) * sigma_r


# def tau_ellipse(temperature=298, viscosity=0.8931e-3, particle_radius_x=4e-6,particle_radius_y=2e-6, particle_radius_z = 0.85e-6, hydro_correction=1.0, sim_length=None, **kwargs):
#     """

#     Computes the diffusive timescale (in SI units) for an anisotropic particle given experimental conditions. Translational diffusion parallel and perpendicular to the principle particle axis is averaged into a single translational diffusion constant as :math:`D^t=(1/2)D_{\\parallel}^t+(1/2)D_{\\perp}^t`. Calculation is modeled after diffusion of anisotropic rods, valid for aspect ratios between 2 and 16, according to `(Yang 2017, J. Chem. Phys.) <https://doi.org/10.1063/1.4995949>`_ and `(Bitter 2017, Langmuir) <https://doi.org/10.1021/acs.langmuir.7b01704>`_. The timescale is computed as:

#     .. math::

#         t = \\frac{D_0^ta_y^2}{D^t\\sigma^2}

#     where :math:`D_0^t` is the simulated diffusivity (unitless, usually 0.25), :math:`D^t` is the average translational diffusivity obtained via measurements or a model (SI), :math:`a_y` is the short axis of the ellipse (SI), and :math:`\\sigma` is the characteristic length scale of the simulation (unitless, usually 1.0)

#     Note that the translational diffusive timescale is also be used as the timescale for rotational dynamics of anisotropic particles.

#     :param temperature: the absolute temperature in [K], defaults to 298K
#     :type temperature: scalar, optional
#     :param viscosity: the viscosity of the medium in [Pa s], defaults to 0.8931e-3
#     :type viscosity: scalar, optional
#     :param particle_radius_x: the radius of the long axis of the ellipse in [m], defaults to 4 microns
#     :type particle_radius_x: scalar, optional
#     :param particle_radius_y: the radius of the short axis of the ellipse in [m], defaults to 2 micron
#     :type particle_radius_y: scalar, optional
#     :param hydro_correction: a (unitless) hydrodynamic correction to the single-particle diffusivity, defaults to 1.0 which has no effect
#     :type hydro_correction: scalar, optional
#     :return: the time it takes for the ellipsoid to diffuse 1 length unit (ay/2) in [s]
#     :rtype: scalar
#     """    
#     # CJY: these expressions should eventually be updated with QW's expressions when they become publishable
    
#     az, ay, ax = np.sort([particle_radius_x, particle_radius_y, particle_radius_z]) 
    
#     if sim_length is None:
#         sim_length = 2*ay

#     s = ax/az
#     r = ax/ay
#     assert s > 1 and r > 1, "Aspect ratios must be greater than 1"
#     wp = ((r-1)*s)/((s-1)*r)
#     wo = 1.0 - wp

#     sigP_t = np.log(s)+(s**2 + 15*s + 18)/(26*s + 8)
#     sigO_t = np.log(s) + (-5*s**2 + 16*s + 30)/(33*s +8)
#     sigma_t = wp * sigP_t + wo * sigO_t

#     sigP_r = np.log(s) + (2.0*s**2 + 13*s - 5.0)/(5*s + 5)
#     sigO_r = np.log(s) + (-2.0*s**2 + 11*s + 12)/(15*s + 6)
#     sigma_r = wp * sigP_r + wo * sigO_r

#     expt_D0_t = hydro_correction * D0_trans(temperature=temperature, viscosity=viscosity, particle_radius=ax) * sigma_t
#     expt_tau_t = (sim_length**2)/(4*expt_D0_t)

#     expt_D0_r = hydro_correction * D0_rot(temperature=temperature, viscosity=viscosity, particle_radius=ax) * sigma_r
#     expt_tau_r = (np.pi / 2.0)**2 / (2.0 * expt_D0_r)

#     return expt_tau_t, expt_tau_r