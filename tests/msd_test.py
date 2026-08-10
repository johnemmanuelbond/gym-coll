import numpy as np
import gsd.hoomd
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from utils import SuperEllipse, quat_to_angle

test_N = 1000
test_DT = 0.7/10.0/4.0 # um^2 / s
test_DR = 0.06/10.0/2.0 # um^2 / s
times = np.linspace(0,10.0,50)

rect = SuperEllipse(ax=4.0,ay=2.0,n=4.0) # dimensions in um

from sims import DynamicMonteCarlo
from sims import BrownianDynamics

def msd(pts, os):
    xs = pts[:,0]
    ys = pts[:,1]
    angles = quat_to_angle(os)
    dist_var = np.mean((xs - np.mean(xs))**2 + (ys - np.mean(ys))**2)
    angl_var = np.mean((angles - np.mean(angles))**2)
    return dist_var, angl_var

class MC(DynamicMonteCarlo):
    def __init__(self, dt):
        super().__init__(dt=dt, DT=test_DT, DR=test_DR, shape=rect)
        self.ideal=True

    @property
    def state(self):
        return msd(self.frame.particles.position, self.frame.particles.orientation)

class BD(BrownianDynamics):
    def __init__(self, dt):
        super().__init__(dt=dt, DT=test_DT, DR=test_DR, shape=rect)
        self.ideal=True

    @property
    def state(self):
        return msd(self.frame.particles.position, self.frame.particles.orientation)

if __name__ == "__main__":
    
    init = gsd.hoomd.Frame()
    init.configuration.box = [100,100,0,0,0,0]
    init.particles.N = test_N
    init.particles.types = ['A']
    init.particles.position = [[0,0,0]]*test_N
    init.particles.orientation = [[1,0,0,0]]*test_N
    init.particles.moment_inertia = [[0,0,10]]*test_N
    init.particles.typeid = np.zeros(test_N,dtype=int)

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(2.0,4.0),dpi=600,sharex=True)

    ax1.set_ylabel("$\\langle\\delta r^2\\rangle / \\mu m^2$")
    ax1.set_ylim([0,1.5])
    ax1.plot(times,4*test_DT*times,linestyle='--',color='k',label=f'$4D_T\\Delta t$')

    ax2.set_ylabel("$\\langle\\delta \\theta^2\\rangle / rad^2$")
    ax2.set_ylim([0,0.1])
    ax2.plot(times,2*test_DR*times,linestyle='--',color='k',label=f'$2D_R\\Delta t$')
    
    ax1.set_xlim([0,times.max()])    
    ax2.set_xlim([0,times.max()])
    ax2.set_xlabel("$\\Delta t/\\tau$")

    for make_sim in [MC,BD]:
        for i, dt in enumerate([1e-1, 1e-2,5e-3]):
            sim = make_sim(dt)
            if isinstance(sim,MC):
                lab = 'mc'
                color='red'
            if isinstance(sim,BD): 
                lab = 'bd'
                color='blue'
            sim.reset(init_state=init,outfile=f"msd_sim_{lab}_dt{dt:.1e}.gsd",nsnap=0.1,mode='wb')
            msds = [sim.state[0]]
            msas = [sim.state[1]]

            start = timer()
            for t in np.diff(times):
                sim.run(t)
                msds.append(sim.state[0])
                msas.append(sim.state[1])
            end = timer()

            ax1.scatter(times,msds,s=10,marker=['o','s','^','H'][i],edgecolors=color,facecolors='none', label=f'{lab} - dt={dt:.1e}')
            ax2.scatter(times,msas,s=10,marker=['o','s','^','H'][i],edgecolors=color,facecolors='none')
            print(f"finished {lab}, dt={dt:.1e}, in {end-start:.2f}s",flush=True)
            fig.savefig("msd.png",bbox_inches='tight')

    ax1.legend(fontsize='x-small')
    ax2.legend(fontsize='x-small')
    fig.savefig("msd.png",bbox_inches='tight')