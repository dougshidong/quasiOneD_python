#! /usr/bin/python3
import sys
import autograd.numpy as np
from numpy.linalg import norm
#from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from numba import jit
import autograd

import sim_data
import mesh

isCompiled=True

# Evaluation of physical quantities
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
def evaluate_p(W, gamma):
    return (gamma - 1.0) * (W[2,:] - (W[1,:]**2/W[0,:])/2)
def evaluate_p1(W1, gamma):
    return (gamma - 1.0) * (W1[2] - (W1[1]**2/W1[0])/2)
def evaluate_primitive_from_state(W, gamma):
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    p = evaluate_p(W, gamma)
    return rho, u, p
def evaluate_all(W, gamma):
    rho, u, p = evaluate_primitive_from_state(W, gamma)
    c = evaluate_c(p, rho, gamma)
    mach = u/c
    return rho, u, p, c, mach

def evaluate_source_state(p, area):
    #Q = np.zeros([3, p.size])
    #Q[1,:] = p * np.diff(area)

    Q = p * np.diff(area)
    return np.array([np.zeros(p.size), Q, np.zeros(p.size)])
def evaluate_convective_state(W, gamma):
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    p = evaluate_p(W, gamma)
    e = W[2,:]

    #F = np.empty(W.shape)
    #F[0,:] = rho
    #F[1,:] = rho*u*u + p
    #F[2,:] = ( e + p ) * u
    F0 = rho
    F1 = rho*u*u + p
    F2 = ( e + p ) * u
    return np.array([F0,F1,F2])

def evaluate_fluxes(W, gamma, scalar_eps):
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    e = W[2,:]
    p = evaluate_p(W, gamma)
    c = evaluate_c(p, rho, gamma)

    F = evaluate_convective_state(W, gamma)

    u_avg = 0.5*(u[:-1]+u[1:])
    c_avg = 0.5*(c[:-1]+c[1:])
    #lamb  = np.max([u_avg + c_avg, u_avg - c_avg], axis=0)
    lamb  = u_avg + c_avg # u is always positive here

    #fluxes = np.empty([3, rho.size+1])
    #fluxes[:,1:-1] = 0.5*((F[:,:-1] + F[:,1:]) - scalar_eps * lamb * (W[:,1:] - W[:,:-1]))
    
    # Returning the fluxes from 1 to n-1 to avoid array assignment for autograd
    fluxes = 0.5*((F[:,:-1] + F[:,1:]) - scalar_eps * lamb * (W[:,1:] - W[:,:-1]))
    return fluxes

def BC_inlet_residual(sim, W_g, W_d, dt, dx):
    R = sim.Cv*(sim.gamma-1.0)
    #r_g = np.copy(W_g[0])
    r_g = W_g[0]
    u_g = W_g[1] / r_g
    p_g = evaluate_p1(W_g, sim.gamma)
    c_g = evaluate_c(p_g, r_g, sim.gamma)

    r_d = W_d[0]
    u_d = W_d[1] / W_d[0]
    p_d = evaluate_p1(W_d, sim.gamma)
    c_d = evaluate_c(p_d, r_d, sim.gamma)

    dW_g = np.zeros(3)
    if u_g < c_g:
        g = sim.gamma
        gm1 = g - 1.0; gp1 = g + 1.0
        dpdu = sim.inlet_total_p * (g / gm1) \
               * (1.0 - (gm1 / gp1) * u_g * u_g / sim.a2)**(1.0 / gm1) \
               * (-2.0 * (gm1 / gp1) * u_g / sim.a2)

        dtdx = dt / dx
        eigenvalue = ((u_g-c_g + u_d-c_d) / 2.0) * dtdx
        dpdx = p_d-p_g
        dudx = u_d-u_g

        du = -eigenvalue * (dpdx - r_g * c_g * dudx) / (dpdu - r_g * c_g)

        u_g_new   = u_g + du
        T_g_new   = sim.inlet_total_T * (1.0 - (gm1 / gp1) * u_g_new * u_g_new / sim.a2)
        p_g_new   = sim.inlet_total_p*(T_g_new / sim.inlet_total_T)**(g / gm1)
        r_g_new   = p_g_new / (sim.R * T_g_new)
        e_g_new   = r_g_new * (sim.Cv * T_g_new + 0.5 * u_g * u_g)

        dW_g[0] = r_g_new           - W_g[0]
        dW_g[1] = r_g_new * u_g_new - W_g[1]
        dW_g[2] = e_g_new           - W_g[2]
    return dW_g

def BC_outlet_residual(sim, W_g, W_d, dt, dx):
    # Returns the update of the ghost vector
    # W_g = Ghost state vector
    # W_d = Domain state vector
    #r_g = np.copy(W_g[0]) # Very important else dW_g = 0
    r_g = W_g[0] # Very important else dW_g = 0
    u_g = W_g[1] / r_g
    p_g = evaluate_p1(W_g, sim.gamma)
    c_g = evaluate_c(p_g, r_g, sim.gamma)

    r_d = W_d[0]
    u_d = W_d[1] / W_d[0]
    p_d = evaluate_p1(W_d, sim.gamma)
    c_d = evaluate_c(p_d, r_d, sim.gamma)

    dtdx = dt/dx
    avgu = 0.5*(u_d+u_g)
    avgc = 0.5*(c_d+c_g)
    eigenvalues0 = avgu * dtdx
    eigenvalues1 = (avgu + avgc) * dtdx
    eigenvalues2 = (avgu - avgc) * dtdx

    dpdx = p_g-p_d
    dudx = u_g-u_d

    Ri0 = -eigenvalues0 * ( (r_g - r_d) - dpdx / c_g**2 )
    Ri1 = -eigenvalues1 * ( dpdx + r_g * c_g * dudx )
    Ri2 = -eigenvalues2 * ( dpdx - r_g * c_g * dudx )

    mach_outlet = avgu / avgc
    dp = 0
    if mach_outlet > 1.0:
        dp = 0.5 * (Ri1 + Ri2)

    drho = Ri0 + dp / c_g**2
    du = (Ri1 - dp) / (r_g * c_g)

    u_g_new = u_g + du
    r_g_new = r_g + drho
    p_g_new = p_g + dp
    T_g_new = p_g_new / (r_g_new * sim.R)
    e_g_new = evaluate_e(r_g_new, T_g_new, u_g_new, sim.Cv)

    dW_g = np.empty_like(W_g)
    dW_g[0] = r_g_new           - W_g[0]
    dW_g[1] = r_g_new * u_g_new - W_g[1]
    dW_g[2] = e_g_new           - W_g[2]
    return dW_g


def evaluate_residual(sim, W, area):
    # Return the residual of the domain 1:n-1
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    p = evaluate_p(W, sim.gamma)
    c = evaluate_c(p, rho, sim.gamma)

    # Returning the fluxes from 1 to n-1 to avoid array assignment for autograd
    fluxes = evaluate_fluxes(W, sim.gamma, sim.scalar_eps)
    Q      = evaluate_source_state(p, area)

    residual = fluxes[:,1:] * (np.ones((3,1))*area[2:-1]) \
               - fluxes[:,:-1] * (np.ones((3,1))*area[1:-2]) \
               - Q[:,1:-1]
#   for i in range(1,rho.size-1):
#       residual[:,i-1] = evaluate_residual_cell(sim, W[:,i-1], W[:,i], W[:,i+1], area[i+1], area[i])
    return residual

#def evaluate_fluxes_twice(Wn, Wp, gamma, scalar_eps):
#    rn = Wn[0,:]
#    un = Wn[1,:] / Wn[0,:]
#    pn = evaluate_p(Wn, gamma)
#    cn = evaluate_c(pn, rn, gamma)
#    Fn = evaluate_convective_state(Wn, gamma)
#
#    rp = Wp[0,:]
#    up = Wp[1,:] / Wp[0,:]
#    pp = evaluate_p(Wp, gamma)
#    cp = evaluate_c(pp, rp, gamma)
#    Fp = evaluate_convective_state(Wp, gamma)
#
#    u_avg = 0.5*(un[:-1]+up[1:])
#    c_avg = 0.5*(cn[:-1]+cp[1:])
#    #lamb  = np.max([u_avg + c_avg, u_avg - c_avg], axis=0)
#    lamb  = u_avg + c_avg # u is always positive here
#
#    # Returning the fluxes from 1 to n-1 to avoid array assignment for autograd
#    fluxes = 0.5*((Fn[:,:-1] + Fp[:,1:]) - scalar_eps * lamb * (Wp[:,1:] - Wn[:,:-1]))
#    return fluxes
#def evaluate_residual_cell(sim, Wn, Wi, Wp, arean, areap):
#    # Return the residual of the domain 1:n-1
#    area = np.array([arean, areap])
#    p = evaluate_p1(Wi, sim.gamma)
#
#    print(Wn)
#    print(Wi)
#    print(Wp)
#    # Returning the fluxes from 1 to n-1 to avoid array assignment for autograd
#    fluxes_m = evaluate_fluxes_twice(Wn, Wi, sim.gamma, sim.scalar_eps)
#    fluxes_p = evaluate_fluxes_twice(Wi, Wp, sim.gamma, sim.scalar_eps)
#    Q      = evaluate_source_state(p, np.array([arean, areap]))
#
#    residual = fluxes_p * (np.ones((3,1))*areap) \
#               - fluxes_n * (np.ones((3,1))*arean) \
#               - Q
#    return residual.T
def update_dt(CFL, dx, W, gamma):
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    p = evaluate_p(W, gamma)
    c = evaluate_c(p, rho, gamma)
    return (CFL * dx) / np.abs(u + c)
def evaluate_dw(sim, W, dx, area):
    dt = update_dt(sim.CFL, dx, W, sim.gamma)

    #dW = np.empty_like(W)
    #new_W = np.copy(W)
    dW = np.empty(W.shape)
    new_W = np.empty(W.shape)
    new_W = W

    for rk_state in range(1,5):
        residual = evaluate_residual(sim, new_W, area)
        new_W[:,1:-1] = new_W[:,1:-1] - (dt[1:-1] / dx[1:-1]) / (5.0 - rk_state) * residual
    dW = (new_W-W)


    #residual = evaluate_residual(sim, W, area)
    #dW[:,1:-1] = -(dt[1:-1] / dx[1:-1]) * residual

    dW[:,0] = BC_inlet_residual(sim, W[:,0], W[:,1], dt[0], dx[0])
    dW[:,-1] = BC_outlet_residual(sim, W[:,-1], W[:,-2], dt[-1], dx[-1])
    #print(dt)
    #print(W+dW)
    #sys.exit()

    return dW

def step_in_time(sim, W, dx, area):
    dW = evaluate_dw(sim, W, dx, area)
    W = W + dW
    return W, np.sum(dW[0,:]**2)
def solve_steady(sim, W, dx, area):
    for flow_iteration in range(sim.iterations_max):
        W, normR = step_in_time(sim, W, dx, area)
        #normR = np.sum((residual[0,:])**2)
        if normR < sim.tolerance: return W
        if flow_iteration%sim.it_print==0: print("Iterations %d \t Residual1 %e" % (flow_iteration, np.sqrt(normR)))
        if(np.isnan(normR)):
            print("\n\nself.W  \n",W)
            break
    return W
def solver(sim):
    # State     variables: rho, rho*u, e
    # Primitive variables: rho, u, p
    # Auxiliary variables: c, mach

    # Create a mesh based on given geometry
    x, dx, xh, area, volume = mesh.initialize_mesh(sim.geom, sim.n_elem)

    # Used initial temperature to initialize primitive variables
    T = isentropic_T(sim.inlet_total_T, sim.inlet_mach, sim.gamma)

    # Initialize primitive variables
    p = np.linspace(sim.inlet_p, sim.outlet_p, sim.n_elem, endpoint=True)
    rho = evaluate_rho(p, T, sim.R)
    c = evaluate_c(p, rho, sim.gamma)
    u = evaluate_u(c, sim.inlet_mach)
    e = evaluate_e(rho, T, u, sim.Cv)

    W = np.empty([3, sim.n_elem])
    W = np.array([rho, rho*u, evaluate_e(rho, T, u, sim.Cv)])
    F = np.empty_like(W)
    Q = np.empty_like(W)


    fluxes = np.empty([3,sim.n_elem+1])

    dt = np.empty_like(volume)
    residual = np.empty_like(W)
    dW = np.empty_like(W)

    W = solve_steady(sim, W, dx, area)
                     
    return W

    
def main():
    sim = sim_data.Simulation_data()
    sim.set_target_geom()

    W = solver(sim)
    x, dx, xh, area, volume = mesh.initialize_mesh(sim.geom, sim.n_elem)
    rho, u, p, c, mach = evaluate_all(W, sim.gamma)
    p_target = p
    area_target = area

    #plt.figure(1)
    #plt.title('Pressure')
    #plt.plot(xh, p_current, label = 'Current')
    #plt.plot(xh, p_target,  label='Target')
    #plt.legend()

    #plt.figure(1)
    #plt.plot(mesh.x, mesh.area,'-o')
    #plt.draw()
    #plt.pause(1)
    #input('Enter to quit')
    #plt.close()
if __name__ == "__main__":
    main()
