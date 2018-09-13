#! /usr/bin/python3
import sys
import numpy as np
from numpy.linalg import norm
#from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import autograd

class Simulation_data:
    def __init__(self, filename = "input.in"):
        self.n_elem         = 10
        self.iterations_max = 60000
        self.it_print       = 20
        self.tolerance      = 1e-14

        self.tolerance      = self.tolerance**2

        self.CFL            = 1.2
        self.scalar_eps     = 0.5

        self.gamma         = 1.4
        self.R             = 1.0

        self.inlet_total_T = 1.0
        self.inlet_total_p = 1.0
        self.inlet_mach    = 1.75
        self.outlet_p      = 0.92

class Mesh:
    def __init__(self, n_elements, x_0 = 0, x_n = 1, parametrization=None):
        self.n_elem = n_elements
        self.n_face = n_elements + 1
        self.x = np.linspace(x_0, x_n, self.n_face, endpoint=True)
        self.dx = np.diff(self.x)
        self.xh = self.x[:-1] + self.dx
        self.area = np.empty_like(self.x)
        self.volume = np.empty_like(self.x)

    
    def evaluate_volume(self, area1, area2, dx):
        return 0.5*(area1+area2)*dx

    def initialize_area_volume(self, a, b, c):
        sine_shape = lambda x, a, b, c: 1 - a * (np.sin(np.pi * x**b))**c
        self.area = sine_shape(self.x, a, b, c)
        self.volume = self.evaluate_volume(self.area[:-1], self.area[1:], self.dx)


def evaluate_rho(p, T, R):
    return p / (R * T)
def evaluate_c(p, rho, gamma):
    return np.sqrt(gamma * p / rho)
def evaluate_u(c, mach):
    return c * mach
def evaluate_mach(u, c): 
    return u/c
def evaluate_e(rho, T, u, Cv):
    return rho * (Cv * T + 0.5 * u*u)
def isentropic_T(total_T, mach, gamma):
    return total_T / (1.0 + (gamma - 1.0) / 2.0 * mach * mach)
def isentropic_p(total_p, mach, gamma):
    return total_p * (1.0 + (gamma - 1.0) / 2.0 * mach*mach)**(-gamma/(gamma-1.0))
def evaluate_e(rho, T, u, Cv ):
    return rho * (Cv * T + 0.5 * u*u)
def evaluate_p(rho, rho_u, e, gamma):
    return (gamma - 1.0) * (e - (rho_u**2/rho)/2)

class Solver:
    def evaluate_primitive_from_state(self, W):
        rho = W[0,:]
        u = W[1,:] / W[0,:]
        p = evaluate_p(W[0,:], W[1,:], W[2,:], self.gamma)
        return rho, u, p

    def __init__(self, mesh, simulation_data):
        # State     variables: rho, rho*u, e
        # Primitive variables: rho, u, p
        # Auxiliary variables: c, mach

        self.sim                  = simulation_data
        self.mesh                 = mesh

        self.gamma                = simulation_data.gamma
        self.R                    = simulation_data.R
        self.Cv                   = self.R / (self.gamma - 1.0);

        self.inlet_mach           = simulation_data.inlet_mach
        self.inlet_total_p        = simulation_data.inlet_total_p
        self.inlet_total_T        = simulation_data.inlet_total_T

        self.a2 = 2.0*self.gamma*self.Cv*self.inlet_total_T \
            *((self.gamma - 1.0) / (self.gamma + 1.0));

        inlet_p         = isentropic_p(self.inlet_total_p, self.inlet_mach, self.gamma)
        self.outlet_p   = simulation_data.outlet_p

        self.p = np.linspace(inlet_p, self.outlet_p, self.mesh.n_elem, endpoint=True)
        T = isentropic_T(self.inlet_total_T, self.inlet_mach, self.gamma)
        self.rho = evaluate_rho(self.p, T, self.R)
        self.c = evaluate_c(self.p, self.rho, self.gamma)
        self.u = evaluate_u(self.c, self.inlet_mach)
        self.e = evaluate_e(self.rho, T, self.u, self.Cv)

        self.dt = np.empty_like(self.mesh.volume)
        self.W = np.empty([3, self.sim.n_elem])
        self.W = np.array([self.rho, self.rho*self.u, evaluate_e(self.rho, T, self.u, self.Cv)])
        self.F = np.empty_like(self.W)
        self.Q = np.empty_like(self.W)

        self.dW = np.empty_like(self.W)
        self.residual = np.empty_like(self.W)

        self.update_dt()
        self.evaluate_convective_state()
        self.evaluate_source_state()

        self.fluxes = np.empty([3,self.mesh.n_elem+1])

    def update_dt(self):
        self.dt = (self.sim.CFL * self.mesh.dx) / np.abs(self.u + self.c)

    def solve_steady(self):
        for flow_iteration in range(self.sim.iterations_max):
            self.step_in_time()

            self.BC_inlet()
            self.BC_outlet()

            normR = np.sum((self.residual[0,:])**2)
            if normR < self.sim.tolerance: return
            if flow_iteration%self.sim.it_print==0: print("Iterations %d \t Residual1 %e" % (flow_iteration, np.sqrt(normR)))
            if(np.isnan(normR)):
                print("\n\nself.W  \n",self.W)
                return
    def step_in_time(self):
        self.evaluate_dw()
        self.W = self.W + self.dW
    def evaluate_dw(self):

        self.evaluate_residual()
        # Update time step based on maximum eigenvalue.
        # Must be done after residual since it uses the primitive variable.
        self.update_dt()

        old_W = self.W
        for rk_state in range(1,5):
            self.evaluate_residual();
            self.W = self.W - (self.dt / self.mesh.volume) / (5.0 - rk_state) * self.residual 

        self.dW = (self.W-old_W)
        self.W = old_W

        #for i in range(3):
        #    self.dW[i,1:-1] = -(self.dt[1:-1] / self.mesh.volume[1:-1]) * self.residual[i,1:-1]

    def evaluate_source_state(self):
        self.Q[0,:] = 0
        self.Q[1,:] = self.p * np.diff(self.mesh.area)
        self.Q[2,:] = 0
    def evaluate_residual(self):
        # Update primitive variable here since it's the deepest the AD will need to go
        #self.rho, self.u, self.p = self.evaluate_primitive_from_state(self.W)
        self.rho = self.W[0,:]
        self.u = self.W[1,:] / self.W[0,:]
        self.p = evaluate_p(self.W[0,:], self.W[1,:], self.W[2,:], self.gamma)
        self.c = evaluate_c(self.p, self.rho, self.gamma)

        self.evaluate_fluxes()
        self.evaluate_source_state()

        self.residual[:,0] = 0
        self.residual[:,-1] = 0
        self.residual[:,1:-1] = self.fluxes[:,2:-1] * (np.ones((3,1))*self.mesh.area[2:-1]) \
                                - self.fluxes[:,1:-2] * (np.ones((3,1))*self.mesh.area[1:-2]) \
                                - self.Q[:,1:-1]

    def evaluate_convective_state(self):
        u = self.u
        u_u = u**2
        rho_u_u = self.W[0,:]*u_u

        self.F[0,:] = self.W[1,:]
        self.F[1,:] = rho_u_u + (self.gamma - 1.0) * (self.W[2,:] - rho_u_u / 2.0)
        self.F[2,:] = ( self.W[2,:] + (self.gamma - 1.0) * (self.W[2,:] - rho_u_u/2.0) ) * u;

    def evaluate_fluxes(self):
        self.c = evaluate_c(self.p, self.rho, self.gamma)
        self.evaluate_convective_state()

        u_avg = 0.5*(self.u[:-1]+self.u[1:])
        c_avg = 0.5*(self.c[:-1]+self.c[1:])
        lamb  = np.max([u_avg + c_avg, u_avg - c_avg], axis=0)

        self.fluxes[:,1:-1] = 0.5*((self.F[:,:-1] + self.F[:,1:]) 
            - self.sim.scalar_eps * lamb * (self.W[:,1:] - self.W[:,:-1]))

    def BC_inlet(self):
        self.rho[1] = self.W[0,1]
        self.u[1] = self.W[1,1] / self.W[0,1]
        self.p[1] = evaluate_p(self.W[0,1], self.W[1,1], self.W[2,1], self.gamma)
        self.c[1] = evaluate_c(self.p[1], self.rho[1], self.gamma)

        if self.u[0] >= self.c[0]:
            self.residual[:,0] = 0.0
            return
        else:
            u0 = self.u[0]; u1 = self.u[1]
            c0 = self.c[0]; c1 = self.c[1]
            g = self.gamma
            gm1 = g - 1.0; gp1 = g + 1.0
            dpdu = self.inlet_total_p * (g / gm1) \
                   * (1.0 - (gm1 / gp1) * u0 * u0 / self.a2)**(1.0 / gm1) \
                   * (-2.0 * (gm1 / gp1) * u0 / self.a2);

            dtdx = self.dt[0] / self.mesh.dx[0];
            eigenvalue = ((u0-c0 + u1-c1) / 2.0) * dtdx;
            dpdx = self.p[1] - self.p[0];
            dudx = u1-u0

            du = -eigenvalue * (dpdx - self.rho[0] * self.c[0] * dudx) / (dpdu - self.rho[0] * self.c[0]);

            u_new = u0 + du
            T_inlet = self.inlet_total_T * (1.0 - (gm1 / gp1) * u_new * u_new / self.a2);

            self.u[0]   = u0 + du;
            self.p[0]   = self.inlet_total_p*(T_inlet / self.inlet_total_T)**(g / gm1);
            self.rho[0] = self.p[0] / (self.R * T_inlet);
            self.e[0]   = self.rho[0] * (self.Cv * T_inlet + 0.5 * self.u[0] * self.u[0]);

            self.residual[0,0] = -(self.rho[0]             - self.W[0,0]) / dtdx;
            self.residual[1,0] = -(self.rho[0] * self.u[0] - self.W[1,0]) / dtdx;
            self.residual[2,0] = -(self.e[0]               - self.W[2,0]) / dtdx;

            self.W[0,0] = self.rho[0];
            self.W[1,0] = self.rho[0] * self.u[0];
            self.W[2,0] = self.e[0];

    def BC_outlet(self):
        #self.rho[-2:], self.u[-2:], self.p[-2:] = self.evaluate_primitive_from_state(self.W[:,-2:])
        self.rho[-2] = self.W[0,-2]
        self.u[-2] = self.W[1,-2] / self.W[0,-2]
        self.p[-2] = evaluate_p(self.W[0,-2], self.W[1,-2], self.W[2,-2], self.gamma)
        self.c[-2] = evaluate_c(self.p[-2], self.rho[-2], self.gamma)

        u0 = self.u[-2]; u1 = self.u[-1]
        c0 = self.c[-2]; c1 = self.c[-1]
        r0 = self.rho[-2]; r1 = self.rho[-1]

        avgu = 0.5*(u0+u1)
        avgc = 0.5*(c0+c1)
        dtdx = self.dt[-1]/self.mesh.dx[-1]
        eigenvalues0 = avgu * dtdx
        eigenvalues1 = (avgu + avgc) * dtdx
        eigenvalues2 = (avgu - avgc) * dtdx

        dpdx = self.p[-1]-self.p[-2]
        dudx = u1-u0

        Ri0 = -eigenvalues0 * ( (r1 - r0) - dpdx / c1**2 )
        Ri1 = -eigenvalues1 * ( dpdx + r1 * c1 * dudx )
        Ri2 = -eigenvalues2 * ( dpdx - r1 * c1 * dudx )

        mach_oulet = avgu / avgc
        if mach_oulet > 1.0:
            dp = 0.5 * (Ri1 + Ri2)
        else:
            dp = 0

        drho = Ri0 + dp / c1**2
        du = (Ri1 - dp) / (r1 * c1)

        self.u[-1] = u1 + du
        self.rho[-1] = r1 + drho
        self.p[-1] = self.p[-1] + dp
        T = self.p[-1] / (self.rho[-1] * self.R)
        self.e[-1] = evaluate_e(self.rho[-1], T, self.u[-1], self.Cv)

        self.residual[0,-1] = (self.W[0,-1] - self.rho[-1]) / dtdx
        self.residual[1,-1] = (self.W[1,-1] - self.rho[-1] * self.u[-1]) / dtdx
        self.residual[2,-1] = (self.W[2,-1] - self.e[-1]) / dtdx

        self.W[0,-1] = self.rho[-1]
        self.W[1,-1] = self.rho[-1] * self.u[-1]
        self.W[2,-1] = self.e[-1]

def main():
    sim_data = Simulation_data()
    mesh = Mesh(sim_data.n_elem)
    mesh.initialize_area_volume(0.10,0.80,6.00)
    
    q1d = Solver(mesh, sim_data)
    q1d.solve_steady()

    #plt.figure(1)
    #plt.plot(mesh.x, mesh.area,'-o')
    #plt.draw()
    #plt.pause(1)
    #input('Enter to quit')
    #plt.close()
if __name__ == "__main__":
    main()
