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


# WIP, POTENTIALLY BEYOND THE SCOPE OF THIS MODULE
# def wall_hydro(use_mean = False, particle_radius=1e-6, debye_length=10e-9, **params):
#     f_para = lambda h: (234*(h/0.5)**2 + 114*(h/0.5) + 2) / (234*(h/0.5)**2 + 241*(h/0.5) + 9)
#     if use_mean:
#         fg = rel_gravity(particle_radius=particle_radius,**params)
#         pf = dlvo_prefactor(particle_radius=particle_radius,**params)
#         kap = kappa(debye_length=debye_length,particle_radius=particle_radius,**params)
#         U_wall = lambda h: 2*pf*np.exp(-kap*h)  - fg*h
#         h = np.linspace(0,2000/kap,100000)    
#         u = U_wall(h)
#         p = np.exp(u.min()-u)
#         return np.average(f_para(h), weights=p)

#     h_min = dlvo_minimum(particle_radius=particle_radius, debye_length=debye_length, **params)
#     return f_para(h_min)


def tau_sphere(temperature=298, viscosity=0.8931e-3, particle_radius=1.0e-6, hydro_correction=0.5, **kwargs):
    """Computes the diffusive timescale (in SI units) for a sphere given experimental conditions

    :param temperature: the absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param viscosity: the viscosity of the medium [Pa s], defaults to 0.8931 mPa s
    :type viscosity: scalar, optional
    :param particle_radius: the radius of the spherical particle in [m], defaults to 1 micron
    :type particle_radius: scalar, optional
    :param hydro_correction: a (unitless) hydrodynamic correction to the single-particle diffusivity, defaults to 0.5 for particle-wall interactions
    :type hydro_correction: scalar, optional
    :return: the time it takes for the sphere to diffuse 1 length unit (2a) in [s]
    :rtype: scalar
    """    
    
    expt_D0 = hydro_correction * kb*temperature/(6*np.pi*viscosity*particle_radius) # with hydro for ptcl-wall
    expt_tau = particle_radius**2/expt_D0
    return expt_tau


def tau_ellipse(temperature=298, viscosity=0.8931e-3, particle_radius_x=4e-6,particle_radius_y=2e-6, hydro_correction=1.0, **kwargs):
    """

    Computes the diffusive timescale (in SI units) for an anisotropic particle given experimental conditions. Translational diffusion parallel and perpendicular to the principle particle axis is averaged into a single translational diffusion constant as :math:`D^t=(1/2)D_{\\parallel}^t+(1/2)D_{\\perp}^t`. Calculation is modeled after diffusion of anisotropic rods, valid for aspect ratios between 2 and 16, according to `(Yang 2017, J. Chem. Phys.) <https://doi.org/10.1063/1.4995949>`_ and `(Bitter 2017, Langmuir) <https://doi.org/10.1021/acs.langmuir.7b01704>`_. The timescale is computed as:

    .. math::

        t = \\frac{D_0^ta_y^2}{D^t\\sigma^2}

    where :math:`D_0^t` is the simulated diffusivity (unitless, usually 0.25), :math:`D^t` is the average translational diffusivity obtained via measurements or a model (SI), :math:`a_y` is the short axis of the ellipse (SI), and :math:`\\sigma` is the characteristic length scale of the simulation (unitless, usually 1.0)

    Note that the translational diffusive timescale is also be used as the timescale for rotational dynamics of anisotropic particles.

    :param temperature: the absolute temperature in [K], defaults to 298K
    :type temperature: scalar, optional
    :param viscosity: the viscosity of the medium in [Pa s], defaults to 0.8931e-3
    :type viscosity: scalar, optional
    :param particle_radius_x: the radius of the long axis of the ellipse in [m], defaults to 4 microns
    :type particle_radius_x: scalar, optional
    :param particle_radius_y: the radius of the short axis of the ellipse in [m], defaults to 2 micron
    :type particle_radius_y: scalar, optional
    :param hydro_correction: a (unitless) hydrodynamic correction to the single-particle diffusivity, defaults to 1.0 which has no effect
    :type hydro_correction: scalar, optional
    :return: the time it takes for the ellipsoid to diffuse 1 length unit (ay/2) in [s]
    :rtype: scalar
    """    
    # AJP: these expressions should eventually be updated with QW's expressions when they become publishable
    ax = max(particle_radius_x,particle_radius_y)
    ay = min(particle_radius_x,particle_radius_y)

    f_para = lambda p: np.log(p) + (-0.4536*p**2 - 1.772*p + 41.5)/(p**2 + 34.38*p + 18.96)
    f_perp = lambda p: np.log(p) + (-0.3604*p**2 + 28.36*p + 72.63)/(p**2 + 36.29*p + 34.9)

    expt_D0 = hydro_correction * kb*temperature/(2*np.pi*viscosity*(ax*2)) * (f_para(ax/ay)/2 + f_perp(ax/ay)/2)
    expt_tau = (ay/2)**2/expt_D0

    return expt_tau


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


def k_coplanar(particle_volume = None, aspect_ratio = None, temperature=298,rel_perm_m=78,voltage=2.0,electrode_gap=100e-6,fcm=-0.4667,**kwargs):
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
    kt = 4*2.45*sign*E
    kr = (6/np.pi)**2*0.3*(1-1/aspect_ratio)*sign*E
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
