# -*- coding: utf-8 -*-
"""
Contains a few helper methods for converting SI values :math:`[m,s,J]` to simulation values :math:`[l,\\tau,kT]`. Most of these functions take in a series of keyword arguments which are best organized into a dictionary, such as :code:`kwargs = units.DEFAULT_PHYSICAL_PARAMETERS`, which can be passed into functions with the unpacking operator, :code:`func(**kwargs)`. The functions below have sensible default values, but kwargs needs to contain the specific keys indicated by the function documentation in order to overwrite the defaults. Importantly, all input arguments *must* have SI units (no prefixes!).
"""

import numpy as np

# physical constants
kb = 1.380e-23 # [J/K] Boltzmann constant
e = 1.602177e-19 # [C] elementary charge
eps = 8.854e-12 # [F/m] or [J/(m*V**2)], permittivity of free space
Na = 6.0221408e23 # [1/mol] Avogadro's number

def calc_fcm(rel_perm_m=78,rel_perm_p=3.2,cond_m=12.6e-4,cond_p=1.52e-4,freq=1e6, **kwargs):
    """
    Calculates the Clausius-Mossotti factor for a spherical colloid under an AC electric field with an applied frequency from material properties particle conductivity, medium conductivity, particle permittivity, medium permittivity. The Clausius-Mossotti factor is used to relate the polarizability of colloidal particles to their dielectric and conductive properties (in a specific medium) `(Pethig 2017, Dielectrophoresis) <https://doi.org/10.1002/9781118671443.ch6>`_.

    .. math::

        f_{cm} = Re\\bigg[\\frac{\\tilde{\\varepsilon_p}-\\tilde{\\varepsilon_m}}{\\tilde{\\varepsilon_p}+2\\tilde{\\varepsilon_m}}\\bigg]

    where :math:`\\varepsilon_p^*` is the complex permittivity of the particle and :math:`\\varepsilon_m^*` is the complex permittivity of the medium, further defined as:

    .. math::

        \\tilde{\\varepsilon_k} = \\varepsilon_k-i \\sigma_k/\\omega

    where :math:`\\varepsilon_k` is the relative permittivity, :math:`\\sigma_k` is the conductivity, and :math:`\\omega` is the frequency of the applied AC field.

    :param rel_perm_m: (unitless) relative permittivity of the medium, defaults to 78 for water
    :type rel_perm_m: scalar, optional
    :param rel_perm_p: (unitless) relative permittivity of the particle, defaults to 3.2 for silica
    :type rel_perm_p: scalar, optional
    :param cond_m: conductivity of the medium in [S/m], defaults to 12.6 uS/cm for water
    :type cond_m: scalar, optional
    :param cond_p: conductivity of the particle in [S/m], defaults to 1.52 uS/cm for silica
    :type cond_p: scalar, optional
    :param freq: the applied AC field frequency in [1/s], defaults to 1 MHz
    :type freq: scalar, optional
    :return: the Clausius-Mossotti factor at the applied frequency
    :rtype: scalar
    """
    # read in parameters with correct unit conversions
    rel_eps_m = rel_perm_m*eps    # [F/m] permittivity of sol'n
    rel_eps_p = rel_perm_p*eps  # [F/m] permittivity of particle
    omega = freq*2*np.pi        # [Hz]

    # Clausius-Mossotti factor from quantities in physical parameters
    ep_p_cplx = rel_eps_p - (cond_p/omega)*complex(0,1)
    ep_m_cplx = rel_eps_m - (cond_m/omega)*complex(0,1)
    fcm = np.real((ep_p_cplx-ep_m_cplx)/(2*ep_m_cplx+ep_p_cplx))

    return fcm


def _Pdf(particle_volume=None,rel_perm_m=78,fcm=-0.4667, **kwargs):
    """
    Calculates the prefactor amplitude of the induced dipole-field interaction potential of particles in an externally applied AC electric field given experimental quantities `(Zhang 2024, Langmuir) <https://doi.org/10.1021/acs.langmuir.4c03101>`_.

    .. math::

        P^{df} = (3/2)\\varepsilon_m v_p f_{cm}
    
    where :math:`\\varepsilon_m` is the permittivity of the medium, :math:`v_p` is the volume of the particle, and :math:`f_{cm}` is the frequency dependent Clausius-Mossotti factor. Note that the medium permittivity is the relative permittivity of the medium and not the complex permittivity of the medium.

    :param particle_volume: volume of the particle in [m], defaults to None
    :type particle_volume: _type_, optional
    :param rel_perm_m: (unitless) relative permittivity of the medium, defaults to 78
    :type rel_perm_m: int, optional
    :param fcm: (unitless) Clausius-Mossotti factor (see above for description), defaults to -0.4667
    :type fcm: float, optional
    :return: the prefactor of the induced dipole-field interaction
    :rtype: scalar
    """
    if particle_volume is None:
        if 'particle_radius_x' in kwargs and 'particle_radius_y' in kwargs:
            particle_volume = np.pi*kwargs['particle_radius_x']*kwargs['particle_radius_y']*kwargs['particle_radius_z']*2
        else:
            particle_volume = 4/3 * np.pi * kwargs['particle_radius']**3
    if fcm is None: fcm = calc_fcm(rel_perm_m=rel_perm_m,**kwargs)
    rel_eps = rel_perm_m*eps
    return 3/2 * particle_volume * rel_eps * fcm


def _E0(voltage=2.0,electrode_gap=100e-6,**kwargs):
    """
    Calculates the root-mean-square (RMS) field amplitude for an externally applied sinusoidal AC electric field. For sinusoidal waveforms, the RMS factor is :math:`\\frac{1}{\\sqrt{2}}`. This is a time-averaged quantity.

    .. math::

        E_0 = \\frac{V}{d_g\\sqrt{8}}
    
    Where :math:`V` is the peak-to-peak voltage of the waveform, and :math:`d_g` is the gap between the electrodes.

    :param voltage: the peak-to-peak voltage of the externally applied field, defaults to 2.0
    :type voltage: float, optional
    :param electrode_gap: electrode gap width, defaults to 100e-6
    :type electrode_gap: float, optional
    :return: RMS field amplitude for a sinusoidal AC field
    :rtype: scalar
    """
    return voltage/electrode_gap/(8**0.5)


def electrode_energy_scale(particle_volume=None,rel_perm_m=78,fcm=-0.4667,voltage=2.0,electrode_gap=100e-6, temperature=293, **kwargs):
    """
    Calculates the prefactor amplitude of the harmonic potential due to the external field given experimental quantities. This prefactor considers both :math:`P^{df}` and the external field parameters in :math:`E_0` to correctly scale the interaction energy based on the electrode geometry:

    .. math::
    
        \\epsilon \\equiv \\frac{-P^{df}E_0^2}{kT} = \\frac{3}{2} v_p \\varepsilon_m f_{cm} \\bigg(\\frac{V}{d_g\\sqrt{8}}\\bigg)^2 \\bigg/ kT

    :param particle_volume: volume of the particle in [m], defaults to None
    :type particle_volume: int, optional
    :param rel_perm_m: (unitless) relative permittivity of the medium, defaults to 78
    :type rel_perm_m: int, optional
    :param fcm: (unitless) Clausius-Mossotti factor, defaults to -0.4667
    :type fcm: float, optional
    :param voltage: peak-to-peak voltage, defaults to 2V
    :type voltage: float, optional
    :param electrode_gap: electrode gap width, defaults to 100e-6
    :type electrode_gap: float, optional
    :param temperature: (K) absolute temperature, defaults to 293
    :type temperature: float, optional
    :return: the prefactor amplitude of the induced dipole-field interaction
    :rtype: scalar
    """
    # eventually, we may want to include local concentration effects (f_eta), either here or in Pdf
    if particle_volume is None:
        if 'particle_radius_x' in kwargs and 'particle_radius_y' in kwargs:
            particle_volume = np.pi*kwargs['particle_radius_x']*kwargs['particle_radius_y']*kwargs['particle_radius_z']*2
        else:
            particle_volume = 4/3 * np.pi * kwargs['particle_radius']**3
    pdf = _Pdf(particle_volume=particle_volume,rel_perm_m=rel_perm_m,fcm=fcm,**kwargs)
    e0  = _E0(voltage=voltage,electrode_gap=electrode_gap,**kwargs)
    kT = kb*temperature

    return -1*pdf*(e0**2)/kT


def k_coplanar(particle_volume = None, aspect_ratio = None, temperature=298,rel_perm_m=78,voltage=2.0,electrode_gap=100e-6,fcm=-0.4667, **kwargs):
    """
    Calculates the prefactors, in [kT] units, on a harmonic external field of form :math:`\\frac{1}{2}k_t (x/d_g)^2 + \\frac{{1}}{{2}}k_r\\cos\\theta` confining a particle to :math:`x=0` and aligning a particle to :math:`\\theta=0` based on experimental condtions. The quantities :math:`k_t` and :math:`k_r` depends on the permittivity of the medium, the volume of the particle (i.e. as calculated from it's three ellipse axes and the superellipse parameter), the Clausius-Mossotti factor for a spherical particle of the same material, the peak-to-peak voltage, and the gap between the coplanar electrodes, as laid out in `Zhang, Langmuir 2024 <https://doi.org/10.1021/acs.langmuir.4c03101?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as>`_.

    :param particle_volume: volume of the confined spherical particle in [m:sup:`3`], defaults to a sphere with a 1 micron radius
    :type particle_volume: scalar, optional
    :param temperature: absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param rel_perm_m: (unitless) permittivity of the medium, defaults to 78 for water
    :type rel_perm_m: scalar, optional
    :param vpp: the applied peak-to-peak voltage across the electrode in [V], defaults to 2V
    :type vpp: scalar, optional
    :param dg: the gap between electrode edges in [m], defaults to 100 microns
    :type dg: scalar, optional
    :param fcm: (unitless) Clausius-Mossotti factor of the particles, defaults to -0.4667
    :type fcm: scalar, optional
    :return: the translational and rotational prefactors on a harmonic external field in [kT]
    :rtype: scalar, scalar
    """
    aspect_ratio = 1.0
    if 'particle_radius_x' in kwargs and 'particle_radius_y' in kwargs:
        ax,ay = (kwargs['particle_radius_x'], kwargs['particle_radius_y'])
        aspect_ratio = max(ax/ay,ay/ax)

    E = electrode_energy_scale(particle_volume=particle_volume,temperature=temperature,
                               rel_perm_m=rel_perm_m,fcm=fcm,electrode_gap=electrode_gap,
                               voltage=voltage, **kwargs)
    sign = np.sign( 0.5 - (voltage<0)*(fcm<0) ) # only need to flip sign if fcm is negative AND vpp is negative
    kt = (4.0 / 3.0) * (3.25 / np.pi) ** 2 * sign * E # for energy matching
    kr = 0.3 * (1.0 - 1.0 / aspect_ratio) * (6.0 / np.pi) ** 2 * sign * E
    return kt, kr


def k_multipole(particle_volume=None,temperature=298,rel_perm_m=78,voltage=2.0,electrode_gap=100e-6,fcm=-0.4667,**kwargs):
    """
    Calculates the prefactor on a 2-dimensional harmonic external field of form :math:`\\frac{1}{2}k (r/d_g)^2` confining a particle to :math:`\\vec{{r}}=(0,0)` in [kT] units based on experimental condtions for a quadrupolar electrode. The quantity :math:`k=32\\epsilon` (with :math:`\\epsilon` returned by :py:meth:`electrode_energy_scale`) depends on the particle volume (i.e. :math:`4/3\\pi a^3`), the temperature, medium permittivity, the applied peak-to-peak voltage, and gap distance between the quadrupolar electrodes.

    :param particle_volume: volume of the confined spherical particle in [m:sup:`3`], defaults to a sphere with a 1 micron radius
    :type particle_volume: scalar, optional
    :param temperature: the absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param rel_perm_m: (unitless) permittivity of the medium, defaults to 78 for water
    :type rel_perm_m: scalar, optional
    :param vpp: applied peak-peak voltage across the electrode in [V], defaults to 2V
    :type vpp: scalar, optional
    :param dg: gap between electrode edges in [m], defaults to 100 microns
    :type dg: scalar, optional
    :param fcm: Claussius-Mossotti factor of particles, defaults to -0.4667
    :type fcm: scalar, optional
    :return: the prefactor on a harmonic external field in [kT]
    :rtype: scalar
    """
    E = electrode_energy_scale(particle_volume=particle_volume,temperature=temperature,
                               rel_perm_m=rel_perm_m,fcm=fcm,electrode_gap=electrode_gap,
                               voltage=voltage,**kwargs)
    sign = np.sign( 0.5 - (voltage<0)*(fcm<0) ) # only need to flip sign if fcm is negative AND vpp is negative
    return 32*(1.028**2)*sign*E
