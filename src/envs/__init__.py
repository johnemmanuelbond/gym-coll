"""Module src/envs/__init__.py."""

from .feedback_control import Discrete
from .feedback_control import Semidiscrete
from .feedback_control import Continuous

from gymnasium.envs.registration import register
register (id="envs/fbc_discrete",entry_point="envs:Discrete")
register (id="envs/fbc_semidiscrete",entry_point="envs:Semidiscrete")
register (id="envs/fbc_continuous",entry_point="envs:Continuous")

# from gymnasium import spaces
# from itertools import product

# def get_list_from_space(space_obj:spaces.Space)->list:
#     """
#     :param space_obj: A gymnasium space with a discrete set of entries, i.e. `Discrete <https://gymnasium.farama.org/api/spaces/fundamental/>`_ or `Tuple <https://gymnasium.farama.org/api/spaces/fundamental/>`_.
#     :type space_obj: `Space <https://gymnasium.farama.org/api/spaces/#gymnasium.spaces.Space>`_
#     :raises NotImplementedError: _description_
#     :return: a list containing all the elements of a gymnasiu space
#     :rtype: list
#     """    
    
#     match type(space_obj):

#         case spaces.Discrete:

#             elements = [int(i + space_obj.start) for i in range(space_obj.n)]
            
#         case spaces.MultiDiscrete:

#             if len(space_obj.nvec.shape)>1:
#                 raise NotImplementedError

#             sets = []
#             for k in range(space_obj.nvec.shape[0]):
#                 sets.append([i  for i in range(space_obj.nvec[k])] )

#             elements = list(product(*sets))
            
#         case spaces.Tuple:

#             if all([ isinstance(obj, spaces.Discrete) for obj in space_obj.spaces]):
            
#                 sets = []
#                 for s in space_obj.spaces:
#                     sets.append([i + s.start for i in range(s.n)])
                                                        
#                 elements = list(product(*sets))

#             else:

#                 raise NotImplementedError
                        
#         case _:

#             raise NotImplementedError

#     return elements
            