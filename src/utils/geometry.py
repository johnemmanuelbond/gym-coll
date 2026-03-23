# -*- coding: utf-8 -*-
"""
Contains a few helper methods for performing common geometrical calculations needed to analyze hoomd simulation data. Includes a class to represent superelliptical shapes.
"""

import numpy as np
import math
from scipy.spatial.distance import cdist

class SuperEllipse():
    """A class to contain helpful methods for characterizing the superellipses used in this module. Superellipses follow the equation:
    
    .. math::

        \\bigg|\\frac{x}{a_x}\\bigg|^n + \\bigg|\\frac{y}{a_y}\\bigg|^n = 1
    
    :param ax: one of the radii of the superellipse, defaults to 1.0
    :type ax: scalar, optional
    :param ay: the other radius of the superellipse, defaults to 1.0
    :type ay: scalar, optional
    :param n: the 'superellipse parameter' defines how sharp the corners are. :math:`n\\to\\infty` produces rectangles, :math:`n=2` gives ellipses, :math:`n\\to1` produces rhombuses, defaults to 2.0:
    :type n: scalar, optional

    .. figure:: media/superellipses.jpg
        :width: 1000
        :height: 140
        :alt: superellipses.jpg: superellipses with varying 'n' at the same aspect ratio 's'
        
        *superellipses with aspect ratio s=2*

    """
    def __init__(self,ax:float=1.0,ay:float=1.0,n:float=2.0):
        """
        Constructor
        """        
        self.ax = max(ax,ay)
        self.ay = min(ax,ay)
        self.n = n
    
    @property
    def aspect(self) -> float:
        """
        :return: the aspect ratio of the superellipse
        :rtype: scalar
        """        
        try:
            return self.ax/self.ay
        except ZeroDivisionError:
            if self.ay==0 and self.ax==0:   return 1
            else: return np.nan
    
    @property
    def area(self) -> float:
        """
        :return: the area of the superellipse
        :rtype: scalar
        """        
        return 4*self.ax*self.ay*(math.gamma(float(1+1/self.n)))**2/(math.gamma(float(1+2/self.n)))
    
    def surface(self,thetas:np.ndarray) -> np.ndarray:
        """returns points along the perimeter of the superellipse associated with polar angles

        :param thetas: an array of polar angles
        :type thetas: ndarray
        :return: the x- and y-positions of points along the perimeter of the superellipse
        :rtype: ndarray
        """        
        rad = self.ax*self.ay * ( np.abs(self.ax*np.sin(thetas))**self.n + np.abs(self.ay*np.cos(thetas))**self.n )**(-1/self.n)
        return np.array([rad*np.cos(thetas), rad*np.sin(thetas)]).T

    def unit_normal(self,thetas:np.ndarray) -> np.ndarray:
        """returns the unit normal vector pointing inward from the superellipse perimeter associated with polar angles

        :param thetas: an array of polar angles
        :type thetas: ndarray
        :return: an array of unit normal vectors associated with the array of polar angles.
        :rtype: ndarray
        """        
        xs,ys = self.surface(thetas).T
        dx1 = np.diff(np.array([*xs,xs[0]]))
        dx2 = np.diff(np.array([xs[-1],*xs]))
        dx = (dx1+dx2)/2

        dy1 = np.diff(np.array([*ys,ys[0]]))
        dy2 = np.diff(np.array([ys[-1],*ys]))
        dy = (dy1+dy2)/2

        dy[dx==0] = np.sign(dy[dx==0])
        dx[dx==0] = 1
        s = dy/dx * np.sign(dx)
        t = -np.ones_like(s)*np.sign(dx)*(dx!=1)
        norm = np.array([s,t])
        
        return (norm/np.linalg.norm(norm,axis=0)).T

    def contact_vertices(self,contact_ratio:float=0.3,n_verts:int=12, require_corners:bool=False):
        """returns an appropriate set of vertices to use in hoomd's `ALJ <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/aniso/alj.html>`_ or `Convexspheropolygon <https://hoomd-blue.readthedocs.io/en/latest/hoomd/hpmc/integrate/convexspheropolygon.html>`_ in order to produce hard-particle behavior associated with this superellipse. To do so simply place smaller particle centers so that they are tangent to the perimeter of the superellipse:

        .. figure:: media/superellipse_contact.jpg
            :width: 1000
            :height: 140
            :alt: superellipse_contact.jpg: small interaction centers are tangent to hard particle perimeter
        
            *The small particles have a "contact radius" diameter by* :math:`\\sigma_c`, *the whole particle has a "core diameter" given by* :math:`\\sigma=2a_y-\\sigma_c`.

        :param contact_ratio: the diameter(/radius) of the smaller contact particle centers, relative to the diameter(/radius) of the hard core, defaults to 0.3
        :type contact_ratio: float, optional
        :param n_verts: the desired number of contact vertices, evenly spaced along the perimeter, defaults to 12
        :type n_verts: int, optional
        :param require_corners: if True, ensures that vertices are placed at the corners of the superellipse, defaults to False
        :type require_corners: bool, optional
        """        
        
        #calculate the interior shell that's one contact sphere radius in from the hard wall
        calc_th = np.linspace(0,2*np.pi*999/1000,1000)
        pts = self.surface(calc_th)
        norm = self.unit_normal(calc_th)
        shell = pts - contact_ratio*self.ay*norm

        #filter out little loopies at extreme ns
        dists = cdist(pts,shell)
        too_close = np.all(dists>=0.99*contact_ratio*self.ay,axis=0)
        shell = shell[too_close]
        calc_th = calc_th[too_close]
        
        wrapped = np.array([shell[-1],*shell,shell[0]])
        tangent_vec = shell[1:]-shell[:-1]
        peri = np.cumsum(np.linalg.norm(tangent_vec,axis=-1))
        evenly_spaced = np.linspace(0,peri[-1],n_verts+1)[:-1]

        #get vertices close to ideal angle and set class properties
        # idx = np.unique(np.array([np.argmin(np.abs(calc_th-t)) for t in th]))
        idx = np.unique(np.array([np.argmin(np.abs(peri-t)) for t in evenly_spaced]))

        if require_corners:
            if self.n > 2:
                phi_corner = np.arctan2(self.ay, self.ax)
                corner_angles = np.array([phi_corner, np.pi-phi_corner, np.pi+phi_corner, 2*np.pi-phi_corner])
            else:
                corner_angles = np.arange(4)*np.pi/2
            corner_idx = np.unique(np.array([np.argmin(np.abs(calc_th-t)) for t in corner_angles]))
            # corner_idx = np.where(np.diff(too_close)<0)[0][:4]
            # print(corner_idx)
            for c in corner_idx:
                idx[np.argmin(np.abs(idx-c))] = c
            # idx = np.unique(np.array([*idx,*corner_idx]))

        self.vertices = np.array([shell[idx][:,0],shell[idx][:,1],np.zeros_like(idx)]).T
        self.core_radius = 2*self.ay*(1-contact_ratio)
        self.contact_ratio = contact_ratio
        self.outsphere = 2*np.linalg.norm(pts,axis=-1).max()


def quat_to_angle(quat:np.ndarray) -> np.ndarray:
    """
    :param quat: a list of quaterions encoding particle orientation
    :type quat: ndarray
    :return: the quadrant-corrected 2d angular orientations of the particles
    :rtype: ndarray
    """    
    angles = 2.0*np.arctan2(quat[:,-1],quat[:,0])
    return angles


def hoomd_box_to_matrix(box:list) -> np.ndarray:
    """returns the matrix form of a hoomd box for use in minimum image calculations

    :param box: a length 6 list of box paramters [Lx,Ly,Lz,xy,xz,yz]
    :type box: array_like
    :return: a matrix containing the basis vectors for the equivalent hoomd box
    :rtype: ndarray
    """    
    return np.array([[box[0],box[3]*box[1],box[4]*box[2]],[0,box[1],box[5]*box[2]],[0,0,box[2]]])


def hoomd_matrix_to_box(box:np.ndarray) -> np.ndarray:
    """returns the hoomd box from a given set of basis vectors

    :param box: a matrix containing the basis vectors of a bounding box
    :type box: ndarray
    :return: a length 6 list of box paramters [Lx,Ly,Lz,xy,xz,yz]
    :rtype: ndarray
    """    
    hbox= np.array([box[0,0],box[1,1],box[2,2],box[0,1]/box[1,1],box[0,2]/box[2,2],box[1,2]/box[2,2]])
    if box[2,2]==0:
        hbox[4]=0
        hbox[5]=0
    return hbox


def minimum_image(coords:np.ndarray, wraps:np.ndarray, basis:np.ndarray) -> np.ndarray:
    """uses the minumum image convention to correctly account for perodic
    boundary conditions when calculating coordinates by using the basis
    vectors of the periodic box.

    :param coords: a [Nxd] list of particle coordinates in d-dimensions
    :type coords: ndarray
    :param wraps: a [Nxd] list of particle *images* in d-dimensions
    :type wraps: ndarray
    :param basis: a [dxd] matrix of basis vectors for the simulation box
    :type basis: ndarray
    :return: a [Nxd] list of particle coordinates corrected for how often they switch images in the periodic boundary conditions
    :rtype: ndarray
    """    
    disp = np.einsum("ij,anj->ani",basis,wraps)
    return coords + disp


def expand_around_pbc(coords:np.ndarray, basis:np.ndarray, padfrac:float = 0.8)->tuple[np.ndarray,np.ndarray]:
    """
    given a frame and a box basis matrix, returns a larger frame which includes
    surrounding particles from the nearest images, as well as the index relating padded
    particles back to their original image. This will enable methods like
    scipy.voronoi to respect periodic boundary conditions.

    :param coords: a [Nxd] list of particle coordinates in d-dimensions
    :type coords: ndarray
    :param basis: a [dxd] matrix of basis vectors for the simulation box
    :type basis: ndarray
    :param padfrac: the number of extra particles, as a fraction of the total number, to include in the 'pad' of surrounding particles, defaults to 0.8
    :type padfrac: float, optional
    :return: a [(N+N*padfrac) x d ] array of particle coordinates in d-dimensions which respect periodic boundary conditions around the central N particles, as well as a [N+N*padfrac] list of indices relating padded particles back to their original image
    :rtype: np.ndarray, np.ndarray
    """    

    pnum = coords.shape[0]
    if basis[2,2]==0: basis[2,2]=1

    frame_basis = (np.linalg.inv(basis) @ coords.T).T
    expanded = np.array([
        *(frame_basis+np.array([ 1, 0, 0])),*(frame_basis+np.array([ 0, 1, 0])),
        *(frame_basis+np.array([-1, 0, 0])),*(frame_basis+np.array([ 0,-1, 0])),
        *(frame_basis+np.array([ 1, 1, 0])),*(frame_basis+np.array([ 1,-1, 0])),
        *(frame_basis+np.array([-1, 1, 0])),*(frame_basis+np.array([-1,-1, 0]))
        ])

    pad_idx = np.argsort(np.max(np.abs(expanded),axis=-1))[:(int(padfrac*pnum))]
    pad = (basis @ expanded[pad_idx].T).T
    
    return np.array([*coords,*pad]), np.array([*np.arange(pnum),*(pad_idx%pnum)])


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    verts = 16
    contact = 0.3
    req_c = True

    # shapes = [SuperEllipse(ax=2,ay=4,n=1), SuperEllipse(ax=2,ay=4,n=2), SuperEllipse(ax=2,ay=4,n=8)]
    shapes = [SuperEllipse(ax=1.0,ay=0.5,n=20),SuperEllipse(ax=1.0,ay=0.5,n=20)]
    thetas = np.linspace(0,2*np.pi,200)+1

    fig,axs = plt.subplots(1,len(shapes),figsize=(1+3*len(shapes),2),dpi=600,layout='tight')

    for s,ax in zip(shapes,axs):
        xs,ys = s.surface(thetas).T
        ax.plot(xs,ys,color='blue')
        ax.text(0,1.15*s.ay,f"n={s.n:.1f}, s={s.aspect:.1f}",fontsize='small',horizontalalignment='center')

        ax.arrow(0,0,s.ax,0,
                 length_includes_head=True,lw=0.5,head_length = 0.2, head_width=0.2,color='k')
        ax.arrow(0,0,-s.ax,0,
                 length_includes_head=True,lw=0.5,head_length = 0.2, head_width=0.2,color='k')
        ax.text(0.1,s.ay-0.3,"$a_x$",fontsize='small',verticalalignment='top',horizontalalignment='left')

        ax.arrow(0,0,0,s.ay,
                 length_includes_head=True,lw=0.5,head_length = 0.2, head_width=0.2,color='k')
        ax.arrow(0,0,0,-s.ay,
                 length_includes_head=True,lw=0.5,head_length = 0.2, head_width=0.2,color='k')
        ax.text(s.ax-0.4,0.05,"$a_y$",fontsize='small',verticalalignment='bottom',horizontalalignment='right')

        ax.axis('off')
        ax.set_aspect('equal')

    fig.savefig('superellipses.jpg')#,bbox_inches='tight')

    fig,axs = plt.subplots(1,len(shapes),figsize=(1+3*len(shapes),2),dpi=600,layout='tight')

    for s,ax, req_c in zip(shapes,axs, [True,False]):
        s.contact_vertices(n_verts=verts,contact_ratio=contact, require_corners=req_c)
        xs,ys = s.surface(thetas).T
        xn,yn = s.unit_normal(thetas).T
        [vx,vy,_] = s.vertices.T
        sig = s.contact_ratio*s.ay
        
        ax.plot(xs,ys,color='blue')
        ax.text(0,1.15*s.ay,f"n={s.n:.1f}, s={s.aspect:.1f}",fontsize='small',horizontalalignment='center')
        ax.plot(xs-sig*xn,ys-sig*yn,color='green',ls='-',lw=0.5)
        
        ax.plot(s.core_radius/2*np.cos(thetas),s.core_radius/2*np.sin(thetas),color='green',ls='--',lw=0.5)
        ax.arrow(0,0,s.core_radius/2,0,
                 length_includes_head=True,lw=0.5,head_length = 0.2, head_width=0.2,color='k')
        ax.arrow(0,0,-s.core_radius/2,0,
                 length_includes_head=True,lw=0.5,head_length = 0.2, head_width=0.2,color='k')
        ax.text(0,0+0.1,"$\\sigma$",fontsize='small',verticalalignment='bottom',horizontalalignment='center')

        for x0,y0 in zip(vx,vy):
            ax.plot(x0+sig*np.cos(thetas),y0+sig*np.sin(thetas),color='green')
        
        vidx = np.argmin(vy)
        ax.arrow(vx[vidx],vy[vidx]-sig-0.1,sig,0,
                 length_includes_head=True,lw=0.5,head_length = 0.2, head_width=0.2,color='k')
        ax.arrow(vx[vidx],vy[vidx]-sig-0.1,-sig,0,
                 length_includes_head=True,lw=0.5,head_length = 0.2, head_width=0.2,color='k')
        ax.text(vx[vidx],vy[vidx]-sig-0.2,"$\\sigma_{{c}}$",fontsize='small',verticalalignment='top',horizontalalignment='center')
    

        ax.axis('off')
        ax.set_aspect('equal')
    

    fig.savefig('superellipse_contact.jpg')#,bbox_inches='tight')
