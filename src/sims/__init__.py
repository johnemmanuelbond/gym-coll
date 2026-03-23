
import importlib.util

from .base import Simbase

from .ldld import General_1D
from .ldld import General_ND

has_hoomd = False
try:
    spec = importlib.util.find_spec('hoomd')
    if spec is not None:
        has_hoomd=True
except ModuleNotFoundError:
    has_hoomd = False

if has_hoomd:
    from .hpmc import Multipole
    from .hpmc import Quadrupole
    from .hpmc import Octopole
    from .hpmc import Coplanar

    from . import hpmc,bd
