# -*- coding: utf-8 -*-
"""
Contains many order parameters for characterizing colloidal ensembles.
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform
from utils.geometry import expand_around_pbc

#first coordination shell for discs at close-packing
DEFAULT_CUTOFF = 0.5*(1+np.sqrt(3))


def neighbors(pts:np.ndarray, neighbor_cutoff:float|None = None, num_closest:int|None = None) -> np.ndarray:
    """Determines neighbors in a configuration of particles based on a cutoff distance.

    :param pts: [Nxd] array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param neighbor_cutoff: specify the distance which defines neighbors. Defaults to halfway between the first coordination peaks for a perfect crystal.
    :type neighbor_cutoff: scalar, optional
    :param num_closest: specify the maximum number of neighbors (within the cutoff) per particle. a.k.a pick that many of the closest neighbors per particle.
    :type num_closest: int, optional
    :return:  [NxN] boolean array indicating which particles are neighbors
    :rtype: ndarray
    """

    dists = squareform(pdist(pts))
    #determines neighbors using a cutoff
    if neighbor_cutoff is None:
        cut = np.ones_like(dists)*DEFAULT_CUTOFF
    else:
        cut = np.ones_like(dists)*neighbor_cutoff
    
    if num_closest is not None:
        cut_i = np.sort(dists,axis=0)[num_closest+2]
        cut = np.array([cut_i,cut]).min(axis=0)

    
    nei = dists < cut
    nei[np.eye(len(pts))>0] = False
    return nei



def stretched_neighbors(pts:np.ndarray, angles:np.ndarray, rx:float = 1.0, ry:float = 1.0, neighbor_cutoff:float = 2.6) ->np.ndarray:
    """Determines neighbors in a configuration of anisotropic particles based on a cutoff distance in the rotated/stretched frame of each particle.

    :param pts: the position of the centers of each anisotropic particle in the configuration
    :type pts: ndarray
    :param angles: the orientation of each anisotropic particle in the configuration
    :type angles: ndarray
    :param rx: the radius of the long axis of the particle (insphere radius times aspect ratio), defaults to 1.0
    :type rx: scalar, optional
    :param ry: the radius of the short axis of the partice (i.e. insphere radius), defaults to 1.0
    :type ry: scalar, optional
    :param neighbor_cutoff: specify the stretched distance which defines neighbors. Defaults to 2.6.
    :type neighbor_cutoff: scalar, optional
    :return: [NxN] boolean array indicating which particles are neighbors
    :rtype: ndarray
    """    

    pnum = pts.shape[0]
    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]
    trig = [np.cos(angles[i]),np.sin(angles[i])]
    xs =  trig[0]*dr_vec[:,:,0]+trig[1]*dr_vec[:,:,1]
    ys = -trig[1]*dr_vec[:,:,0]+trig[0]*dr_vec[:,:,1]

    # format into NxN boolean array
    nei = np.sqrt((xs/rx)**2+(ys/ry)**2) < neighbor_cutoff
    nei[np.eye(len(pts))>0] = False

    return nei


def padded_neighbors(pts, basis, cutoff=DEFAULT_CUTOFF, padfrac=0.8):
    """Determines neighbors in a configuration of particles based on a cutoff distance while respecting the periodic boundary condition using :py:meth:`utils.geometry.expand_around_pbc`.

    :param pts: [Nxd] array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param basis: a [dxd] matrix of basis vectors for the simulation box
    :type basis: ndarray
    :param neighbor_cutoff: specify the distance which defines neighbors. Defaults to halfway between the first coordination peaks for a perfect crystal.
    :type neighbor_cutoff: scalar, optional
    :param padfrac: the number of extra particles, as a fraction of the total number, to include in the 'pad' of surrounding particles, defaults to 0.8
    :return:  [NxN] boolean array indicating which particles are neighbors
    :rtype: ndarray
    """

    pnum = pts.shape[0]
    pts_padded, idx_padded = expand_around_pbc(pts,basis,padfrac=padfrac)
    
    nei_padded = squareform(pdist(pts_padded)) <= cutoff    
    nei_padded[np.eye(pts_padded.shape[0])==1]=False
    nei_real = nei_padded[:pnum,:pnum]
    for i,n in enumerate(nei_padded[:pnum]):
        nei_real[i][np.unique(idx_padded[n])] = True

    return nei_real


def bond_order(pts:np.ndarray, nei_bool:np.ndarray|None = None, order:int = 6) -> tuple[np.ndarray,float]:
    """Calculates the local and global bond orientational order parameter of each particle in a 2D configuration with respect to the y axis. The local n-fold bond orientaitonal order for a particle :math:`j` is:

    .. math::

        \\psi_j = \\frac{1}{N_j}\\sum_k\\psi_{jk}=\\frac{1}{N_j}\\sum_ke^{in\\theta_{jk}}

    Where the sum is over all :math:`N_j` neighboring particles :math:`k` to particle :math:`j` and :math:`\\theta_{jk}` is the angle between particles :math:`j` and :math:`k`. Similarly, the global n-fold bond orientational order is:

    .. math::

        \\psi_g = \\frac{1}{N}\\frac{1}{N_j}\\sum_{jk}\\psi_{jk}=\\frac{1}{N}\\frac{1}{N_j}\\sum_{jk}e^{in\\theta_{jk}}

    Where the sum is over all unique bonds between particles :math:`j` and :math:`k`.

    :param pts: [Nxd] array of particle positions in 'd' dimensions, though the calculation only access the first two dimensions.
    :type pts: ndarray
    :param nei_bool: a [NxN] boolean array indicating neighboring particles, will calculate neighbors using the default cutoff value if none is given.
    :type nei_bool: ndarray, optional
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :return: [N] array of complex bond orientational order parameters, and the norm of their mean.
    :rtype: tuple(ndarray, scalar)
    """

    pnum = pts.shape[0]
    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]
    if nei_bool is None:
        nei_bool = squareform(pdist(pts))<=DEFAULT_CUTOFF
        nei_bool[i==j]=False

    # double-check for degenerate neighborless states
    if not np.any(nei_bool): return np.zeros(pnum),0
    # get neighbor count per particle, and cumulative
    sizes = np.sum(nei_bool,axis=-1)
    csizes = np.cumsum(sizes)

    #assemble lists of vectors to evaluate angles between (us and vs)
    bonds = dr_vec[nei_bool]
    xs = bonds[:,0]
    ys = bonds[:,1]
    angles = np.arctan2(ys,xs)
    
    # now we compute psi_ij for each of the bonds
    psi_ij = np.exp(1j*order*angles)
    # pick out only the last summed psi for each particle
    psi_csum = np.array([0,*np.cumsum(psi_ij)])
    # subtract off the previous particles' summed psi values
    c_idx = np.array([0,*csizes])
    psi = psi_csum[c_idx[1:]]-psi_csum[c_idx[:-1]]

    #return the neighbor-averaged psi
    psi[sizes>0]*=1/sizes[sizes>0]
    psi[sizes==0]=0
    
    return psi, np.abs(np.mean(psi_ij))


def stretched_bond_order(pts:np.ndarray, angles:np.ndarray, nei_bool:np.ndarray|None = None, rx:float = 1.0, ry:float = 1.0, order:int = 6) -> tuple[np.ndarray,float]:
    """Computes the local and global stretched bond orientational order parameter. This calculation rotates coordinates into a frame of reference stretched according to the long and short axes of each particle according to equations given in `(Torrez-Diaz Soft Matter, 2022) <https://doi.org/10.1039/D1SM01523K>`_.

    :param pts: [Nxd] array of particle positions in 'd' dimensions, though the calculation only access the first two dimensions.
    :type pts: ndarray
    :param angles: the orientation of each anisotropic particle in the configuration
    :type angles: ndarray
    :param nei_bool: a [NxN] boolean array indicating neighboring particles, will calculate stretched neighbors using the default cutoff value if none is given.
    :type nei_bool: ndarray, optional
    :param rx: the radius of the long axis of the particle (insphere radius times aspect ratio), defaults to 1.0
    :type rx: scalar, optional
    :param ry: the radius of the short axis of the partice (i.e. insphere radius), defaults to 1.0
    :type ry: scalar, optional
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :return: [N] array of complex bond orientational order parameters, and the norm of their mean.
    :rtype: tuple(ndarray, scalar)
    """    
    pnum = pts.shape[0]
    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]
    if nei_bool is None:
        nei_bool = stretched_neighbors(pts,angles,rx=rx,ry=ry)

    # double-check for degenerate neighborless states
    if not np.any(nei_bool): return np.zeros(pnum),0
    # get neighbor count per particle, and cumulative
    sizes = np.sum(nei_bool,axis=-1)
    csizes = np.cumsum(sizes)

    #assemble lists of vectors to evaluate angles between (us and vs)
    bonds = dr_vec[nei_bool]
    orient = angles[i[nei_bool]]
    trig = [np.cos(orient), np.sin(orient)]

    xs =  trig[0]*bonds[:,0]+trig[1]*bonds[:,1]
    ys = -trig[1]*bonds[:,0]+trig[0]*bonds[:,1]
    angles = np.arctan2(ys/ry,xs/rx) + orient

    # now we compute psi_ij for each of the bonds
    psi_ij = np.exp(1j*order*angles)
    # pick out only the last summed psi for each particle
    psi_csum = np.array([0,*np.cumsum(psi_ij)])
    # subtract off the previous particles' summed psi values
    c_idx = np.array([0,*csizes])
    psi = psi_csum[c_idx[1:]]-psi_csum[c_idx[:-1]]

    #return the neighbor-averaged psi
    psi[sizes>0]*=1/sizes[sizes>0]
    psi[sizes==0]=0
    
    return psi, np.abs(np.mean(psi_ij))


def gyration_radius(pts:np.ndarray) -> float:
    """returns the radius of gyration according to the formula

    .. math::

        R_g^2 \\equiv \\frac{1}{N}\\sum_k|\\mathbf{r}_k-\\bar{\\mathbf{r}}|^2 = \\frac{1}{N^2}\\sum_{j>i}|\\mathbf{r}_i - \\mathbf{r}_j|^2 

    where the :math:`j>i` in the summation index indicates that repeated pairs are not summed over

    :param pts: [Nxd] array of particle positions in 'd' dimensions
    :type pts: ndarray
    :return: the radius of gyration of the particles about their center of mass
    :rtype: scalar
    """    
    N = len(pts)
    dists = pdist(pts)
    Rg2 = np.sum(dists**2) / (N**2) # pdist accounts for the factor of 2 in the denominator
    return np.sqrt(Rg2)


def gyration_tensor(pts:np.ndarray, ref:np.ndarray|None = None) -> np.ndarray:
    """returns the gyration tensor (for principal moments analysis) of an ensemble of particles according to the formula

    .. math::

        S_{mn} = \\frac{1}{N}\\sum_{j}r^{(j)}_m r^{(j)}_n

    where the positions, :math:`r`, are defined in their center of mass reference frame

    :param pts: [Nxd] array of particle positions in 'd' dimensions,
    :type pts: ndarray
    :param ref: point in d-dimensional space from which to reference particle positions, defaults to the mean position of the points. Use this for constraining the center of mass to the surface of a manifold, for instance.
    :type ref: ndarray , optional
    :return: the [dxd] gyration tensor of the ensemble
    :rtype: ndarray
    """    

    if ref is None:
        ref = pts.mean(axis=0)
    assert (pts.shape[-1],) == ref.shape, 'reference must have same dimesionality as the points'
    centered = pts - ref
    gyrate = centered.T @ centered
    return gyrate/len(pts)

def circularity(gyr_tensor:np.ndarray=None, pts=None) -> float:
    """
    the 'circularity' of a colloidal cluster as used in `Zhang, Sci. Adv. 2020 <https://doi.org/10.1126/sciadv.abd6716>`_. This metric is calcuated using the principal moments of the :py:meth:`gyration_tensor`. For 2d ensembles, after diagonalization:

    .. math::

        S_{mn} = \\begin{pmatrix}
        \\lambda_1^2 & 0 & 0 \\\\
        0 & \\lambda_2^2 & 0 \\\\
        0 & 0 & \\lambda_3^2=0
        \\end{pmatrix}

    With the prinicipal moments defined as :math:`\\lambda_1>\\lambda_2`. Under these definitions, we can compute the radius of gyration :math:`R_g=\\sqrt{{\\lambda_1^2 + \\lambda_2^2}}` and the acylindricity :math:`a = \\lambda_1-\\lambda_2`, then combine them to define the circularity:

    .. math::
        c = 1 - \\frac{a}{R_g}
    
    When the cluster is circular the two principal moments are equal and so :math:`a=0 \\to c=1`. When the cluster is a linear chain, the smaller principal moment approaches zero, and so :math:`a=R_g=\\lambda_1 \\to c=0`.

    :param gyr_tensor: [dxd] array , defaults to None
    :type gyr_tensor: ndarray, optional
    :param pts: an ensemble of colloidal positions, defaults to None
    :type pts: ndarray, optional
    :return: _description_
    :rtype: float
    :raises AssertionError: if neither the gyration tensor nor an ensemble of points are supplied
    """

    assert (gyr_tensor is not None) or (pts is not None), "User must supply either the gyration tensor or the ensemble of points needed to calculate it"

    if gyration_tensor is None:
        Smn = gyration_tensor(pts)
    else:
        Smn = gyr_tensor

    mom, _ = np.linalg.eig(Smn)
    lx = np.sort(mom)[-1]
    ly = np.sort(mom)[-2]

    a = lx**0.5 - ly**0.5
    rg = (lx+ly)**0.5
    c = 1-a/rg

    return c

def global_nematic(angles):
    """
    Computes the global nematic order of a particle ensemble.

    .. math::

        S_2 = max_{\\theta_2}\\langle\\cos(2(\\theta_j-\\theta_2))\\rangle = |\\langle e^{2i\\theta_j}\\rangle_j|

    :param angles: the orientation of each particle in the frame
    :type angles: ndarray
    :return: the value of the global nematic order and the angle which defines it
    :rtype: scalar, scalar
    """

    s2 = np.exp(2j*angles).mean()
    return np.abs(s2), np.angle(s2)/2

def local_nematic(angles, nei_bool = None, pts = None):
    """
    Computes the local nematic order, per particle, of an ensemble. Quantifies the local orientational order of a system by calculating a director for particle i via its neighboring particle(s) j, as defined by `(Baron J. Chem. Phys. 2023) <https://doi.org/10.1063/5.0169659>`_.

    .. math::

        S_{2,i} = max_{\\theta_{2,i}}\\langle\\cos(2(\\theta_j-\\theta_{2,i}))\\rangle_{j(r_{ij}<6a_x)} = |\\langle e^{2i\\theta_j}\\rangle_{j(r_{ij}<6a_x})|

    :param angles: the orientation of each particle in the frame
    :type angles: ndarray
    :param nei_bool: [NxN] boolean array defining particle neighbors, defaults to all particles within 6a\\ :sub:`x` for an ellipse with a\\ :sub:`y` = 0.5 and a\\ :sub:`x` = 1.0
    :type nei_bool: ndarray, optional
    :param pts: the positions of each particle in the frame
    :type pts: ndarray, optional
    :return: [N] array of the local nematic parameter and the associated angle of the director
    :rtype: ndarray, ndarray
    """
    if nei_bool is None:
        assert not (pts is None), "supply particle positions for neighbor calculation" 
        nei_bool = neighbors(pts,neighbor_cutoff=6)

    nei_angles = np.full(nei_bool.shape,angles)*nei_bool
    nb = nei_bool.sum(axis=-1)
    s2 = (np.exp(2j*nei_angles)*nei_bool).sum(axis=-1)/nb
    s2[nb==0]=0+0j

    return np.abs(s2), np.angle(s2)/2

def global_tetratic(angles):
    """
    Computes the global tetratic order of a particle ensemble.

    .. math::

        T_4 = max_{\\theta_4}\\langle\\cos(4(\\theta_j-\\theta_4))\\rangle = |\\langle e^{4i\\theta_j}\\rangle_j|

    :param angles: the orientation of each particle in the frame
    :type angles: ndarray
    :return: the value of the global tetratic order and the angle which defines it
    :rtype: scalar, scalar
    """

    t4 = np.exp(4j*angles).mean()
    return np.abs(t4), np.angle(t4)/4

def crystal_connectivity(psis:np.ndarray, nei_bool:np.ndarray, crystallinity_threshold:float = 0.32, norm:float|None=None) -> np.ndarray:
    """Computes the crystal connectivity of each particle in a 2D configuration. The crystal connectivity measures the similarity of bond-orientational order parameter over all pairs of neighboring particles in order to determine which particles are part of a definite crystalline domain. The crystal connectivity of particle :math:`j` is given by:

    .. math::
        
        C_n^j = \\frac{1}{n}\\sum_k^{\\text{nei}}\\bigg[ \\frac{\\text{Re}\\big[\\psi_j\\psi_k^*\\big]}{|\\psi_j\\psi_k^*|} \\geq \\Theta_C \\bigg]

    Where the :math:`\\psi`\'s are any-fold bond orientational order parameters for each particle, :math:`\\Theta_{C}` is a \'crystallinity threshold\' used to determine whether two neighboring particles are part of the same crystalline domain, and :math:`n` is a factor used to simply normalize :math:`C_6^j` between zero and one.


    :param psis: [N] array of complex bond orientational order parameters
    :type psis: ndarray
    :param nei_bool: a [NxN] boolean array indicating neighboring particles.
    :type nei_bool: ndarray,
    :param crystallinity_threshold: the minimum innier product of adjacent complex bond-OPs needed in order to consider adjacent particles 'connected', defaults to 0.32
    :type crystallinity_threshold: float, optional
    :param norm: an optional factor to normalize the result, defaults to the connectivity value for a perfectly crystalline hexagon (i.e. equations 8 and 10 in the SI from `(Juarez, Lab on a Chip 2012) <https://doi.org/10.1039/C2LC40692F>`_)
    :type norm: float | None, optional
    :return: [N] array of real crystal connectivities
    :rtype: ndarray
    """    

    # double-check for degenerate neighborless states
    if not np.any(nei_bool): return np.zeros_like(psis)

    if norm is None:
        # computing the reference c6 for a perfect lattice
        # equation 8 from SI of https://doi.org/10.1039/C2LC40692F
        shells = -1/2 + np.sqrt((len(psis)-1)/3 + 1/4)
        # equation 10 from SI of https://doi.org/10.1039/C2LC40692F
        c6_hex = 6*(3*shells**2 + shells)/len(psis)
        norm = c6_hex

    psi_prod = np.outer(psis,np.conjugate(psis))
    chi_ij   = np.abs(np.real(psi_prod))
    chi_ij[np.abs(psi_prod)>0]*= 1/np.abs(psi_prod)[np.abs(psi_prod)>0]

    c6 = np.sum( (chi_ij*nei_bool)>=crystallinity_threshold, axis=-1)

    return c6/norm