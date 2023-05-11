from compass_gait_walker import CompassGaitModel
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root

class PassiveWalker:
    def __init__(self,walker):
        self.walker = walker
        
    def fixed_point(self,zg):
        zstar = fsolve(self.poincare_map, zg,full_output=True) 
        return zstar[0]

    
    def poincare_map(self,z):
        t = 0
        [t1,z1] = self.walker.one_gait_step(t,z)
        N = len(t1)-1
        return [z1[N,0]-z[0], z1[N,1]-z[1],z1[N,2]-z[2],z1[N,3]-z[3]]
    
    def impact_map_jacobian(self,z):
        pass
    
if __name__ == '__main__':
    model = CompassGaitModel()
    q1 = 0.2; u1 = -0.25; q2 = -0.4; u2 = 0.2
    z0 = np.array([q1,u1,q2,u2])
    passive_walker = PassiveWalker(model)
    zstar=passive_walker.fixed_point(z0)
    model.render_mode = 1
    model.log_enable = 1
    [t,z] = model.simulate(0,zstar,2)
    model.logger.plot_data()
    