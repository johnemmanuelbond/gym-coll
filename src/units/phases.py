# -*- coding: utf-8 -*-
"""
Contains a few helper methods for converting SI values :math:`[m,s,J]` to simulation values :math:`[l,\\tau,kT]`. Most of these functions take in a series of keyword arguments which are best organized into a dictionary, such as :code:`kwargs = units.DEFAULT_PHYSICAL_PARAMETERS`, which can be passed into functions with the unpacking operator, :code:`func(**kwargs)`. The functions below have sensible default values, but kwargs needs to contain the specific keys indicated by the function documentation in order to overwrite the defaults. Importantly, all input arguments *must* have SI units (no prefixes!).
"""

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

def get_a_eff(phi,debye_points = None):
    """
    From an isotropic interaction potential in simulation units (2a, kT), returns the effective radius for a hard disc interaction with the same second virial coeffecient as the interaction potential.

    :param phi: a function which takes in a length (2a-scale) and returns an energy (kT-scale)
    :type phi: function
    :param debye_points: A list of points to explicitly include in the integration (for numerical precision reasons), defaults to None
    :type debye_points: array_like, optional
    :return: the effective radius of particles interacting with phi
    :rtype: scalar
    """

    if debye_points is None:
        debye_points = np.linspace(0,0.15,5)

    integrand = lambda r: 1-np.exp(-1*phi(r+1))
    
    first, fErr = quad(integrand, 0, debye_points[-1], points=debye_points)
    second, sErr = quad(integrand, debye_points[-1], np.inf)

    return (0.5 + 1/2*(first+second))


def phase_boundaries(aspect_ratio=1.0,superellipse_param=2.0, *kwargs):
    """
    Returns the approximate phase boundaries for a given aspect ratio and superellipse parameter based on empirical fits to simulation data from `(Zhang, J. Chem. Phys. 2024) <https://doi.org/10.1063/5.0238904>`_. The returned values are the approximate volume fractions for the nematic (eta_n) < freezing (eta_f) < melting (eta_m) < close-packed (eta_cp) transitions.

    :param aspect_ratio: the aspect ratio of the superellipse, defaults to 1.0
    :type aspect_ratio: scalar, optional
    :param superellipse_param: the superellipse parameter, defaults to 2.0
    :type superellipse_param: scalar, optional
    :return: the approximate volume fractions for the nematic (eta_n), freezing (eta_f), melting (eta_m), and close-packed (eta_cp) transitions
    :rtype: scalar, scalar, scalar, scalar
    """
    s = aspect_ratio
    n = superellipse_param
    if n == 2:
        eta_cp = 0.907
    else:
        eta_cp = gamma(1+1/n)**2/(gamma(1+2/n))

    if n == 2 and s == 1:
        eta_f = 0.70#0.69    # freezing point
        eta_m = 0.715#0.71     # melting point
        eta_n = eta_f

    else:
        _s_demo = np.arange(2.6,8.1,0.1)
        _eta_f_demo = 0.5*_s_demo**-6.57 + 0.818
        eta_f = np.interp(s, [1,*_s_demo], [0.686,*_eta_f_demo])
        _eta_m_demo = 0.1*_s_demo**-3.50 + 0.833
        eta_m = np.interp(s, [1,*_s_demo], [0.723,*_eta_m_demo])

        if n == 2 and s <= 2:
            eta_n = 0.6
        elif (s > 1 and s <= 1.6) or (n>2 and s == 1):
            eta_n = eta_f
        else:
            eta_n = 6.37/(5.14+s+4)

    return eta_n, eta_f, eta_m, eta_cp


def vx_qpole(pnum=100,debye_length=30e-9,**kwargs):
    """For a given particle number, calculates the voltage needed to bring all particles into one crystal based on experiments from `(Zhang, J. Chem. Phys. 2024) <https://doi.org/10.1063/5.0238904>`_.

    :param pnum: number of particles, defaults to 100
    :type pnum: int, optional
    :param debye_length: the electrostatic screening length of the medium in [m], defaults to 30 nm
    :type debye_length: scalar, optional
    :return: voltage required to crystallize \'pnum\' particles
    :rtype: scalar
    """    

    l = debye_length*1e9
    a = lambda l: 7.15 + 4.10e-3 * l
    b = lambda l: 0.219 + 4.24e-4 * l
    return  a(l)*(pnum**(-b(l)))


def veta_bpole(eta0, eta_cp=0.907, particle_area = None, pnum=100, box_height = 100e-6, electrode_gap = 100e-6, **kwargs):
    """For a given initial volume fraction, calculates the voltage needed to bring all particles into one crystal based on experiments from `(Edwards, Soft Matter 2013) <https://doi.org/10.1039/C3SM50809A>`_.

    :param eta0: initial volume fraction of particles
    :type eta0: scalar
    :param eta_cp: close-packed volume fraction of particles, defaults to 0.907
    :type eta_cp: scalar, optional
    :param particle_area: cross-sectional area of the particle in [m:sup:`2`], defaults to a \\~3 micron disc
    :type particle_area: scalar, optional
    :param pnum: number of particles, defaults to 100
    :type pnum: int, optional
    :param box_height: height of the sample cell in [m], defaults to 100 microns
    :type box_height: scalar, optional
    :param electrode_gap: gap between electrode edges in [m], defaults to 100 microns
    :type electrode_gap: scalar, optional
    :return: voltage required to assemble particles from an initial volume fraction of \'eta0\'
    :rtype: scalar
    """

    if particle_area is None:
        if 'particle_radius_x' in kwargs and 'particle_radius_y' in kwargs:
            particle_area = np.pi*kwargs['particle_radius_x']*kwargs['particle_radius_y']
        else:
            particle_area = np.pi * kwargs['particle_radius']**2

    deta = eta0 - eta_cp
    g = 1e-3*(-0.6/deta + 9*deta + 3.7) # V

    E0 = box_height*electrode_gap/pnum * particle_area**-1.5 * g # V/m
    # return E0
    return E0*(8**0.5) * electrode_gap # V
