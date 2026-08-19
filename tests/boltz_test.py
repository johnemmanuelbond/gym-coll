import numpy as np
import gsd.hoomd
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from units.ac_field import _Pdf, _E0, kb, eps

import hoomd
from hoomd.dep.units import k_multipole
from hoomd.dep import ForceQuadrupole, ExternalQuadrupole
from hoomd.dep import ForceOctupole, ExternalOctupole
from hoomd.dep import ForceAnypole, ExternalAnypole

from sims import DynamicMonteCarlo
from sims import BrownianDynamics

test_N = 1000
eq_time = 500
electrode_gap = 100 # um

phys = {
    "temperature": 298,
    "rel_perm_m": 78,
    "ion_multiplicity": 1,
    "debye_length": 30.0e-09,
    "viscosity": 0.0008931,
    "particle_radius": 1.435e-06,
    "particle_density": 1980,
    "surface_potential": -50.0e-03,
    "fcm": -0.4667,
    "fps": 8,
    }

phys['electrode_gap'] = electrode_gap*1e-6
k_ref = k_multipole(voltage=1.0, **phys)

def Er(r, electrode_gap=100e-6, **kwargs):
    return _E0(**kwargs) * 4 * r / electrode_gap * (1.028 + (1.961e-5)*r + (8.341e-5)*r**2 - (1.539e-9)*r**3 + (2.081e-7)*r**4)

def lam(rel_perm_m=78,particle_radius=3e-6,temperature=298,fcm=-0.5,**kwargs):
    eps_m = eps*rel_perm_m
    a = particle_radius
    kT = kb*temperature
    return np.pi*eps_m * (a**3) * (fcm*_E0(**kwargs))**2 / kT

def U_expt(rs, fcm=-0.5, **kwargs):
    return -2*lam(fcm=fcm,**kwargs) * (Er(rs,**kwargs)/_E0(**kwargs))**2 / fcm


class MC_Qpole(DynamicMonteCarlo):
    def __init__(self, voltage):
        super().__init__(dt=1e-1)
        self.ideal=True
        qpole = ExternalQuadrupole(electrode_gap=electrode_gap)
        for t in self.types:
            qpole.params[t] = dict(k=voltage**2 * k_ref)
        self.externals = [qpole]


class BD_Qpole(BrownianDynamics):
    def __init__(self, voltage):
        super().__init__(dt=1e-1)
        self.ideal=True
        qpole = ForceQuadrupole(electrode_gap=electrode_gap)
        for t in self.types:
            qpole.params[t] = dict(k=voltage**2 * k_ref)
        self.forces = [qpole]


class MC_Opole(DynamicMonteCarlo):
    def __init__(self, voltage, q0):
        super().__init__(dt=1e-1)
        self.ideal=True
        opole = ExternalOctupole(electrode_gap=electrode_gap, electrode_orientation=q0)
        for t in self.types:
            opole.params[t] = dict(k_para=(voltage/1.3)**2 * k_ref,
                                   k_perp=(1.3*voltage)**2 * k_ref)
        self.externals = [opole]


class BD_Opole(BrownianDynamics):
    def __init__(self, voltage, q0):
        super().__init__(dt=1e-1)
        self.ideal=True
        opole = ForceOctupole(electrode_gap=electrode_gap, electrode_orientation=q0)
        for t in self.types:
            opole.params[t] = dict(k_para=(voltage/1.3)**2 * k_ref,
                                   k_perp=(1.3*voltage)**2 * k_ref)
        self.forces = [opole]

# class MC_Qpole(DynamicMonteCarlo):
#     def __init__(self, voltage):
#         super().__init__(dt=1e-1)
#         self.ideal=True
#         qpole = ExternalQuadrupole(electrode_gap=electrode_gap)
#         for t in self.types:
#             qpole.params[t] = dict(k_trans=voltage**2 * k_ref,
#                                     k_rot=0.0,
#                                     m_sym=0)
#         self.externals = [qpole]


# class BD_Qpole(BrownianDynamics):
#     def __init__(self, voltage):
#         super().__init__(dt=1e-1)
#         self.ideal=True
#         qpole = ForceQuadrupole(electrode_gap=electrode_gap)
#         for t in self.types:
#             qpole.params[t] = dict(k_trans=voltage**2 * k_ref,
#                                    k_rot=0.0,
#                                    m_sym=0)
#         self.forces = [qpole]

if __name__ == "__main__":

    init_x = np.random.uniform(-electrode_gap/2,electrode_gap/2,test_N)*0.65
    init_y = np.random.uniform(-electrode_gap/2,electrode_gap/2,test_N)*0.65
    init_z = np.zeros(test_N)
    rand_t = np.random.uniform(-np.pi,np.pi,test_N)*0.9
    
    init = gsd.hoomd.Frame()
    init.configuration.box = [electrode_gap,electrode_gap,0,0,0,0]
    init.particles.N = test_N
    init.particles.types = ['A']
    init.particles.position = np.array([init_x,init_y,init_z]).T
    init.particles.orientation = np.array([np.cos(rand_t/2),np.zeros(test_N),np.zeros(test_N),np.sin(rand_t/2)]).T
    init.particles.image = np.zeros((test_N,3))
    init.particles.moment_inertia = [[0.0,0.0,10.0]]*test_N
    init.particles.typeid = np.zeros(test_N,dtype=int)

    fig,ax = plt.subplots(1,1,figsize=(2.0,2.0),dpi=600)

    ax.set_xlim([0,1/4])
    ax.set_xticks([0,1/12,1/6,1/4])
    ax.set_xticklabels(['$0$','$1/12$','$1/6$','$1/4$'])
    ax.set_xlabel("$r/d_g$")
    ax.set_ylabel('$U/kT$')
    ax.set_ylim([0,10])

    for i, v in enumerate([1,2]):

        xx = np.linspace(0,electrode_gap/2,100)
        label = "$\\lambda f_{{CM}}^{{-1}}|E(r)/E_0|^2$" if i == 0 else None
        ax.plot(xx/electrode_gap,U_expt(xx*1e-6,voltage=v,**phys),linestyle='--',color='k', label=label)

        for make_sim in [MC_Qpole,BD_Qpole]:
            sim = make_sim(v)
            if isinstance(sim,DynamicMonteCarlo):
                lab = 'mc'
                color='red'
            if isinstance(sim,BrownianDynamics): 
                lab = 'bd'
                color='blue'

            start = timer()
            sim.reset(init_state=init)
            sim.run(eq_time)
            frame = sim.microstate
            sim.reset(init_state=frame,outfile=f"boltz_qpole_{lab}_v{v:.1f}.gsd",nsnap=1.0,mode='wb')
            sim.run(eq_time)
            end = timer()

            frames = gsd.hoomd.open(f"boltz_qpole_{lab}_v{v:.1f}.gsd",mode='r')
            pts = np.array([f.particles.position for f in frames])
            rs = np.linalg.norm(pts,axis=-1).flatten()
            edges = np.linspace(0,electrode_gap/2,100)
            bin_areas = np.pi*(edges[1:]**2 - edges[:-1]**2)
            counts, _ = np.histogram(rs,bins=edges,density=False)
            p = counts/bin_areas
            mids = 0.5*(edges[1:] + edges[:-1])
            U_trans = -np.log(p/p.max())
            ax.scatter(mids/electrode_gap,U_trans,s=10,marker=['o','s'][i],edgecolors=color,facecolors='none', label=f'{lab} - {v:.1f}V')

            frames.close()

            print(f"finished {lab}, V={v:.1f}, in {end-start:.2f}s",flush=True)
            fig.savefig("boltz-qpole.png",bbox_inches='tight')

    ax.legend(fontsize='xx-small')
    fig.savefig("boltz-qpole.png",bbox_inches='tight')




    fig,ax = plt.subplots(1,1,figsize=(2.0,2.0),dpi=600)
    
    ax.set_xlim([-1/4,1/4])
    ax.set_xticks([-1/4,-1/8,0,1/8,1/4])
    ax.set_xticklabels(['$-1/4$','$-1/8$','$0$','$1/8$','$1/4$'])
    ax.set_xlabel("$x/d_g$")
    ax.set_ylim([-1/4,1/4])
    ax.set_yticks([-1/4,-1/8,0,1/8,1/4])
    ax.set_yticklabels(['$-\\frac{{1}}{{4}}$','$-\\frac{{1}}{{8}}$','$0$','$\\frac{{1}}{{8}}$','$\\frac{{1}}{{4}}$'])
    ax.set_ylabel("$\\frac{{y}}/{{d_g}}$", rotation=90)

    xx = np.linspace(-electrode_gap/2,electrode_gap/2,101)
    yy = np.linspace(-electrode_gap/2,electrode_gap/2,101)
    XX, YY = np.meshgrid(0.5*(xx[1:]+xx[:-1]), 0.5*(yy[1:]+yy[:-1]))

    for i, q0 in enumerate([np.pi/4,3*np.pi/4]):

        voltage = [0.4,2.0][i]
        k_para = (voltage/1.3)**2 * k_ref
        k_perp = (1.3*voltage)**2 * k_ref
        levels = np.array([0,1,2,3,4,5])

        sinq = np.sin(q0)
        cosq = np.cos(q0)
        dr_para = (XX*cosq + YY*sinq)/electrode_gap
        dr_perp = (-XX*sinq + YY*cosq)/electrode_gap
        U_theory = 0.5*k_para*dr_para**2 + 0.5*k_perp*dr_perp**2

        # label = "$\\lambda f_{{CM}}^{{-1}}|E(r)/E_0|^2$" if i == 0 else None
        label = ["RH conic","LH conic"][i]
        ax.contour(XX[0]/electrode_gap,YY[:,0]/electrode_gap,U_theory,linestyle=['-','--'][i],colors='k', label=label,levels=levels,lw=0.7)

        for make_sim in [MC_Opole,BD_Opole]:
            sim = make_sim(voltage, q0)
            if isinstance(sim,DynamicMonteCarlo):
                lab = 'mc'
                color='red'
            if isinstance(sim,BrownianDynamics): 
                lab = 'bd'
                color='blue'

            start = timer()
            sim.reset(init_state=init)
            sim.run(eq_time)
            frame = sim.microstate
            sim.reset(init_state=frame,outfile=f"boltz_opole_{lab}_{i}.gsd",nsnap=1.0,mode='wb')
            sim.run(eq_time)
            end = timer()

            frames = gsd.hoomd.open(f"boltz_opole_{lab}_{i}.gsd",mode='r')
            pts = np.array([f.particles.position for f in frames])

            counts = np.histogram2d(pts[:,:,0].flatten(),pts[:,:,1].flatten(),bins=100,range=[[-electrode_gap/2,electrode_gap/2],[-electrode_gap/2,electrode_gap/2]],density=False)[0]
            p = counts.T/np.sum(counts)
            U_boltz = -np.log(p/p.max())
            ax.contour(XX[0]/electrode_gap,YY[:,0]/electrode_gap,U_boltz,levels=levels,colors=color, linestyle=["-","--"][i], label=f'{lab} - {["RH","LH"][i]}', lw=0.3)
            frames.close()

            print(f"finished {lab}, {i}, in {end-start:.2f}s",flush=True)
            fig.savefig("boltz-opole.png",bbox_inches='tight')

    # ax.legend(fontsize='xx-small')
    fig.savefig("boltz-opole.png",bbox_inches='tight')