# -*- coding: utf-8 -*-
"""
The base class for colloidal simulations within the gym-coll framework.

Simulations use chemical physics to evolve a configuration of particles (or representation therof). A simulation must be able to run (i.e. evolve the state) and reset (return to initial conditions) for use within a gymnasium environment. 

The state of a simulation is characterized by a vector of order parameters which characterize it's properties.
"""

class Simbase:
    """
    The base class for colloidal simulations.
    This class is meant to be inherited by a specific simulation and shouldn't be instantiated directly.
    """

    def __init__(self):
        """
        Constructor method
        """
        pass


    @property
    def state(self) -> tuple:
        """
        :return: the position of the simulation in order parameter space
        :rtype: tuple
        """        
        return (None,)

    @property
    def elapsed(self)-> int:
        """
        :return: the total number of run calls since the last reset
        :rtype: int
        """
        return self.step
    
    @property
    def dims(self) -> int:
        """
        :return: the dimensionality of the simulation in order parameter space
        :rtype: int
        """        
        return len(self.state)

    def run(self, span, action):
        """
        Runs the simulation for a short burst under some external condition

        :param span: the length of the short burst
        :param action: a global thermodynamic variable which modulates the behavior of the simulation
        """        
        self.step+=1

    def reset(self, **kwargs):
        """
        Resets the simulation back to initial conditions
        """
        self.step=0

