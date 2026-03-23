# -*- coding: utf-8 -*-
"""
Multiple classes for general low-dimensional langevin dynamics (LDLD) simulations. In LDLD simulations a low-dimensional vector of order parameters defines the state of a system. Defining the free energy over low-D space provides a driving force to evolve the state through low-D space. Defining the diffusion tensor over low-D space provides a drag force to evolve the state through low-D space. These two forces, plus a thermalized random force, provide the basis for a low-dimensional equation of motion with which to model, then characterize, dynamics.

There are various ways to define the FEL (free energy landscape) and the DL (diffusion landscape). In the simplest formulation  constant scalar diffusion is sufficient to capture dynamics, and only the bottom 5kT of a quadratic FEL is necessary. However, this module supports more complex formulations and can generally support any callable FEL and DL.

Unit definitions for LDLD require some explanation. Energy units are always defined by :math:`kT`, the thermal energy scale, however length and time units are LDLD units are often abstract and heavily user defined.

The LDLD module may be used to simulate particles directly, in which case length and time units can be particle diameters (:math:`2a`) and mean free times (:math:`\\tau`) or, more directly, nanometers and seconds. In these cases diffusion landscapes are specified in length squared per time units as usual.

However, LDLD is more commonly used to simulate (unitless) order parameter evolution, in which case there *is no* length unit, and consequently the diffusion landscape must only have inverse time units. This more abstract diffusion landscape, in turn, defines the time unit for LDLD. Since order parameter diffusion landscapes are most often measured (as opposed to defined *a priori*), the time unit of an LDLD simulation most often refers to the time unit of whatever experiment or simulation generated its diffusion landscape.

WIP: add source(s)
"""
import numpy as np

from sims import Simbase




class General_1D(Simbase):
    """
    Evolves a one-dimensional coordinate, :math:`x(t)` according to the langevin equation:

    .. math::
        x(t+\\delta t) = x(t) - D(x | a)dt\\frac{1}{kT}\\nabla U(x | a) + \\nabla D(x | a)dt + \\Gamma\\sqrt{D(x | a) dt}
    
    Breaking this down, the random variable :math:`\\Gamma` is a uniformly distributed  with mean zero and variance one in order to simulate thermal noise on the scale of the energy landscape :math:`U/kT`. :math:`a` is a external thermodynamic variable which modulates the free energy and diffusion landscapes. The free energy landscape, :math:`U(x|a)`, is defined over all values of the coordinate :math:`x` and under some external condition :math:`a`. Its gradient along the coordinate defines a driving force towards lower free energies. The diffusion landscape, :math:`D(x|a)`, is similarly defined over all values of :math:`x` and under some condition :math:`a`. Its value controls the relative size of displacements due to thermal noise (:math:`\\sim\\Gamma\\sqrt{D}`) and displacements due to the driving force (:math:`\\sim D\\nabla U`).

    WIP: explain :math:`\\nabla D` (easier said than done). For readers, it's often in the don't-worry-about-it space.

    :param FEL: :math:`U(x | a)` is the free energy landscape in :math:`kT` energy units. Must be a callable function of the simulation's current position, :math:`x(t)`, and an external thermodynamic variable, :math:`a`, (such as a voltage or pressure).
    :param DL: :math:`D(x | a)` is the diffusion landscape in abstract units. Must be a callable function of the simulation's current position, :math:`x(t)`, and an external thermodynamic variable, :math:`a`
    :param kT: the temperature in simulation energy units, defaults to 1
    :type kT: scalar, optional
    :param dt: the size of the timestep in simulation time units, defaults to 1e-2
    :type dt: scalar, optional
    :param dx: the width along the FEL and DL, in simulation length units, with which to numerically calculate gradients, defaults to 1e-5
    :type dx: scalar, optional
    :param x_max: the maximum value the coordinate is allowed to take in simulaiton length units, defaults to 1.0
    :type x_max: scalar, optional
    :param seed: the seed for the random number generator on :math:`\\Gamma`, defaults to None
    :type seed: int | None, optional
    """
    
    def __init__(self,
                 FEL,
                 DL,
                 kT: float = 1,
                 dt: float = 1e-2,
                 dx: float = 1e-5,
                 x_max: float = 1.0,
                 seed:int | None = None):
        """
        Constructor
        """        
        self._LD = {"dt":dt, "kT":kT, "xm":x_max}
        super().__init__()
        self.reset(seed=seed)

        self.diff_functional = DL
        self.energy_functional = FEL
        self.force_functional = lambda x,a: -(self.energy_functional(x+0.5*dx,a) - self.energy_functional(x-0.5*dx,a))/dx


    @property
    def dims(self) -> int:
        """
        :return: the dimensionality of the simulation
        :rtype: int
        """        
        assert len(self.state) == 1
        return 1
    
    @property
    def state(self) -> tuple:
        """
        :return: The position of the simulation in low-dimensional space
        :rtype: tuple
        """        
        return (self.x,)
    
    @property
    def elapsed(self) -> float:
        """
        :return: the total elasped time (in simulation units) since the last reset
        :rtype: scalar
        """
        return self.t

    @property
    def dt(self)-> float:
        """
        :return: integration timestep in simulation units
        :rtype: scalar
        """        
        return self._LD['dt']
    
    @property
    def kT(self) -> float:
        """
        :return: the temperature in simulation energy units
        :rtype: scalar
        """        
        return self._LD['kT']
    
    @kT.setter
    def kT(self,kT:float):
        """
        :param kT: the temperature in simulation energy units
        :type kT: scalar
        """        
        self._LD['kT']=kT

    @property
    def max(self) -> float:
        """
        :return: the maximum allowed value of the coordinate to stay within observation space
        :rtype: scalar
        """        
        return self._LD['xm']
    
    @max.setter
    def max(self,x_max:float):
        """
        :param x_max: the maximum allowed value of the coordinate to stay within observation space
        :type x_max: scalar
        """        
        self._LD['xm']=x_max


    def reset(self,
              x0: float | np.ndarray | None = None,
              seed:int | None = None):
        """
        Resets the simulation, reinitializes the configuration at a random point. Starting the simulation from a numpy array will run many independent simulations in parallel under the same external condition specfied in :py:func:`run`.

        :param x0: an initial condition of one (or many parallel) simulation, defaults to random initialization
        :type x0: scalar | ndarray | None, optional
        :param seed: the random number generator seed used for thermal fluctuations, defaults to None
        :type seed: int | None, optional
        """        
        super().reset()

        self._rng = np.random.default_rng(seed=seed)
        xm = self.max
        if x0 is None:
            self.x = self._rng.random()*xm
        else:
            self.x = np.clip(x0,0,xm)
        self.t = 0

    def run(self,
            time: float,
            action:float):
        """Steps the simulation forward for a short burst under an external condition. Uses the `midpoint algorithm <https://doi.org/10.1017/S0022112095000176>`_ for a variable diffusion landscape.
        
        :param time: the runtime in simulation time units of this short simulation burst
        :type time: scalar
        :param action: the thermodynamic varible that controls the low-dimensional free energy landscape and diffusion landscape
        :type action: scalar
        """        
        super().run(time,action)
        
        kT = self.kT
        dt = self.dt
        xm = self.max
        steps = int(time/dt)
        d = np.array(self.state).flatten().size

        for _ in range(steps):
            # gamma = self._rng.normal(loc=0.0,scale=1.0)
            gamma = np.sqrt(12)*(self._rng.random(d)-0.5)
            
            F1 =  self.force_functional(self.x,action)
            D1  =  self.diff_functional(self.x,action)
            mid_x = self.x + 0.5*(D1*dt/kT)*F1 + 0.5*np.sqrt(2*D1*dt)*gamma
            mid_x = np.clip(mid_x,0,xm)

            F2 = self.force_functional(mid_x,action)
            D2 = self.diff_functional(mid_x,action)
            new_x = self.x + (D2*dt/kT)*F2 + np.sqrt(2*(D2**2/D1)*dt)*gamma
            self.x = np.clip(new_x,0,xm)

            new_t = self.t + dt
            self.t = new_t



class General_ND(Simbase):
    """
    Evolves a d-dimensional coordinate, :math:`\\mathbf{x}(t)` according to the langevin equation in d dimensions:

    .. math::
        x_i(t+\\delta t) = x_i(t) - D_{ij}(\\mathbf{x} | a)dt\\frac{1}{kT}\\nabla_j U(\\mathbf{x} | a) + \\nabla_j D_{ij}(\\mathbf{x} | a)dt + \\Gamma_j\\sqrt{D_{ij}(\\mathbf{x} | a) dt}
    
    This equation is written using `einstein notation <https://en.wikipedia.org/wiki/Einstein_notation>`_ where repeated indices are summed over (i.e. matrix multiplication). Indices on vectors like :math:`x_i` run from 1 to d, as do the two indices on the tensor :math:`D_{ij}`. The current position of the simulation may be represented as a vector :math:`\\mathbf{x}=(x_1,...,x_d)`. As before, the vector-valued random variable :math:`\\Gamma_j` is a uniformly distributed  with mean zero and variance one in order to simulate thermal noise on the scale of the energy landscape :math:`U/kT`. :math:`a` is a external thermodynamic variable which modulates the free energy and diffusion landscapes. The scalar-valued free energy landscape, :math:`U(\\mathbf{x}|a)`, is defined over all values of each coordinate :math:`x_i` and under some external condition :math:`a`. Its gradient along *each* coordinate defines a driving force towards lower free energies *in that coordinate*. The tensor-valued diffusion landscape, :math:`D_{ij}(\\mathbf{x}|a)`, is similarly defined over all values of each :math:`x_i` and under some condition :math:`a`. Its value controls the relative size of displacements due to thermal noise (:math:`\\sim\\Gamma_j\\sqrt{D_{ij}}`) and displacements due to the driving force (:math:`\\sim D_{ij}\\nabla_j U`) *in each coordinate*.

    WIP: explain :math:`\\nabla_j D_{ij}` (easier said than done). For readers, it's often in the don't-worry-about-it space.

    :param dims: dimensionality, d, of this low-d simulation.
    :type dims: int
    :param FEL: :math:`U(\\mathbf{x} | a)` is the free energy landscape in :math:`kT` energy units. Must be a callable function of the simulation's current vector-valued position, :math:`\\mathbf{x}(t)`, and an external thermodynamic variable, :math:`a`, (such as a voltage or pressure). It must output a simple scalar. (for vectorized functions it should output a list of scalars indexed by each input :math:`\\mathbf{x}`)
    :param DL: :math:`D_{ij}(\\mathbf{x} | a)` is the diffusion landscape in abstract units. Must be a callable function of the simulation's current vector valued position, :math:`\\mathbf{x}(t)`, and an external thermodynamic variable, :math:`a`. It must output a matrix of shape [dxd] (for vectorized functions it should output a list of matrices indexed by each input :math:`\\mathbf{x}`).
    :param kT: the temperature in simulation energy units, defaults to 1
    :type kT: scalar, optional
    :param dt: the size of the timestep in simulation time units, defaults to 1e-2
    :type dt: scalar, optional
    :param dx: the width along the FEL and DL, in simulation length units, with which to numerically calculate gradients, defaults to 1e-5
    :type dx: scalar, optional
    :param x_max: the maximum value the coordinate is allowed to take in simulaiton length units, defaults to 1.0
    :type x_max: scalar, ndarray, optional
    :param seed: the seed for the random number generator on :math:`\\Gamma`, defaults to None
    :type seed: int | None, optional
    """
    
    def __init__(self,
                 dims: int,
                 FEL,
                 DL,
                 kT: float = 1,
                 dt: float = 1e-2,
                 dx: float = 1e-5,
                 x_max: float | np.ndarray = 1.0,
                 seed:int | None = None):
        """
        Constructor
        """        
        self._d = dims
        if type(x_max) is float:
            xm = np.ones(dims)*x_max
        else:
            xm = x_max
        self._LD = {"dt":dt, "kT":kT, "xm":xm}
        super().__init__()
        self.reset(seed=seed)

        self.diff_functional = DL
        self.energy_functional = FEL
        self.force_functional = lambda x,a: np.array([-(self.energy_functional(x+0.5*dv,a) - self.energy_functional(x-0.5*dv,a))/dx for dv in dx*np.eye(self._d)]).T

    @property
    def dims(self) -> int:
        """
        :return: the dimensionality of the simulation
        :rtype: int
        """        
        assert self._d == len(self.state)
        return self._d

    @property
    def state(self) -> tuple:
        """
        :return: The position of the simulation in low-dimensional space, *indexed by coordinate*.
        :rtype: tuple
        """        
        if len(self.x) == 1:
            return tuple(self.x.flatten().tolist())
        else:
            return tuple([xi for xi in self.x.T])
    
    @property
    def elapsed(self) -> float:
        """
        :return: the total elasped time (in simulation units) since the last reset
        :rtype: scalar
        """
        return self.t

    @property
    def dt(self)-> float:
        """
        :return: integration timestep in simulation units
        :rtype: scalar
        """        
        return self._LD['dt']
    
    @property
    def kT(self) -> float:
        """
        :return: the temperature in simulation energy units
        :rtype: scalar
        """        
        return self._LD['kT']
    
    @kT.setter
    def kT(self,kT:float):
        """
        :param kT: the temperature in simulation energy units
        :type kT: scalar
        """        
        self._LD['kT']=kT

    @property
    def max(self) -> np.ndarray:
        """
        :return: the maximum allowed value of each coordinate to stay within observation space
        :rtype: ndarray
        """        
        return self._LD['xm']
    
    @max.setter
    def max(self,x_max:float | np.ndarray):
        """
        :param x_max: the maximum allowed value of each coordinate to stay within observation space (if given a scalar, assume it's the max for each dimension)
        :type x_max: scalar, ndarray
        """        
        if type(x_max) is float:
            xm = np.ones(self._d)*x_max
        else:
            xm = x_max
        self._LD['xm']=xm


    def reset(self,
              x0: np.ndarray | None = None,
              seed:int | None = None):
        """
        Resets the simulation, reinitializes the configuration at a random point. Starting the simulation from a numpy array will run many independent simulations in parallel under the same external condition specfied in :py:func:`run`.

        :param x0: an initial condition of one (or many parallel) simulation states (a numpy array of shape [d], or a list thereof), defaults to random initialization for one simulation.
        :type x0: ndarray | None, optional
        :param seed: the random number generator seed used for thermal fluctuations, defaults to None
        :type seed: int | None, optional
        """        
        super().reset()

        self._rng = np.random.default_rng(seed=seed)
        xm = self.max
        if x0 is None:
            self.x = np.array([self._rng.random(self._d)*xm])
        else:
            try:
                x0 = np.array(x0)
            except:
                raise Exception("initial state is not formattable into numpy array")
        
            if x0.shape[-1] != self.dims: raise Exception("Initial state has incorrect dimensions")
        
            if   len(x0.shape)==1:
                self.x = np.array([np.clip(x0,0*xm,xm)])
            elif len(x0.shape)==2:
                self.x = np.clip(x0,0*xm,xm)
            else:
                raise Exception("Please flatten initial state")
        
        self.t = 0

    def run(self,
            time: float,
            action:float):
        """Steps the simulation forward for a short burst under an external field. Uses the `midpoint algorithm <https://doi.org/10.1017/S0022112095000176>`_ for a variable diffusion landscape.
        
        :param time: the runtime in simulation time units of this short simulation burst
        :type time: scalar
        :param action: the thermodynamic varible that controls the low-dimensional free energy landscape and diffusion landscape
        :type action: scalar
        """        
        super().run(time,action)
        
        kT = self.kT
        dt = self.dt
        xm = self.max
        steps = int(time/dt)

        for _ in range(steps):
            # gamma = self._rng.normal(loc=0.0,scale=1.0)
            gamma = np.sqrt(12)*(self._rng.random(self.x.shape)-0.5)
            
            F1 =  self.force_functional(self.x,action)
            D1  =  self.diff_functional(self.x,action)
            mid_x = self.x + 0.5*np.matvec((D1*dt/kT),F1) + 0.5*np.matvec(np.sqrt(2*D1*dt),gamma)
            mid_x = np.clip(mid_x,0*xm,xm)

            F2 = self.force_functional(mid_x,action)
            D2 = self.diff_functional(mid_x,action)
            D1[D2==0] = np.inf
            D1[D1==0] = np.inf
            new_x = self.x + np.matvec((D2*dt/kT),F2) + np.matvec(np.sqrt(2*(D2**2/D1)*dt),gamma)
            self.x = np.clip(new_x,0*xm,xm)

            new_t = self.t + dt
            self.t = new_t



if __name__ == "__main__":
    pass
