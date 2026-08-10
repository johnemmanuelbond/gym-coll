"""Module src/units/__init__.py."""

# physical constants
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



from .diffusivity import sphere_trans_diff, sphere_rot_diff, tau_sphere
from .diffusivity import shape_trans_diff, shape_rot_diff

from .units import dlvo_prefactor
from .units import kappa

from .ac_field import electrode_energy_scale, k_coplanar, k_multipole

from .phases import get_a_eff, phase_boundaries, vx_qpole, veta_bpole