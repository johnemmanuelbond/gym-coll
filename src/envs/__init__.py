"""Module src/envs/__init__.py."""

from .feedback_control import Discrete
from .feedback_control import Semidiscrete
from .feedback_control import Continuous

from gymnasium.envs.registration import register
register (id="envs/fbc_discrete",entry_point="envs:Discrete")
register (id="envs/fbc_semidiscrete",entry_point="envs:Semidiscrete")
register (id="envs/fbc_continuous",entry_point="envs:Continuous")

