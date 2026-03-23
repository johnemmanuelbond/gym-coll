import importlib.util




from .geometry import SuperEllipse
from .geometry import quat_to_angle
from .geometry import hoomd_box_to_matrix
from .geometry import hoomd_matrix_to_box
from .geometry import minimum_image
from .geometry import expand_around_pbc




has_hoomd = False
try:
    spec = importlib.util.find_spec('hoomd')
    if spec is not None:
        has_hoomd=True
except ModuleNotFoundError:
    has_hoomd = False

if has_hoomd:
    from .hoomd_helpers import Electrodes
    from .hoomd_helpers import random_frame
    from .hoomd_helpers import hoomd_dlvo, capped_dlvo, hoomd_wca, hoomd_alj, hpmc_dipoles
    # from .hoomd_helpers import TypeUpdater
    # from .hoomd_helpers import DLVO_table




from .gym_spaces import get_list_from_space