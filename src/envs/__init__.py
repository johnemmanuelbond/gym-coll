"""Module src/envs/__init__.py."""

from .feedback_control import obsD_actD
from .feedback_control import obsC_actD
from .feedback_control import obsC_actC

from gymnasium.envs.registration import register
register (id="envs/fbc_discrete",entry_point="envs:obsD_actD")
register (id="envs/fbc_semidiscrete",entry_point="envs:obsC_actD")
register (id="envs/fbc_continuous",entry_point="envs:obsC_actC")

