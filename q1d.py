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


#@jit(nopython=isCompiled)
def evaluate_volume(area1, area2, dx):
    return 0.5*(area1+area2)*dx
#@jit(nopython=isCompiled)
def initialize_area_volume(a, b, c):
    return area, volume
def initialize_mesh(geom, n_elem, x_0 = 0, x_n = 1, parametrization=None):
    n_elem = n_elem
    n_face = n_elem + 1
    x = np.linspace(x_0, x_n, n_face, endpoint=True)
    dx = np.diff(x)
    xh = x[:-1] + dx
    sine_shape = lambda x, a, b, c: 1 - a * (np.sin(np.pi * x**b))**c
    area = sine_shape(x, geom[0], geom[1], geom[2])
    volume = evaluate_volume(area[:-1], area[1:], dx)
    return x, dx, xh, area, volume

# Evaluation of physical quantities
#@jit(nopython=isCompiled)
def evaluate_rho(p, T, R):
    return p / (R * T)
#@jit(nopython=isCompiled)
def evaluate_c(p, rho, gamma):
    return np.sqrt(gamma * p / rho)
#@jit(nopython=isCompiled)
def evaluate_u(c, mach):
    return c * mach
#@jit(nopython=isCompiled)
def evaluate_mach(u, c): 
    return u/c
#@jit(nopython=isCompiled)
def evaluate_e(rho, T, u, Cv):
    return rho * (Cv * T + 0.5 * u*u)
#@jit(nopython=isCompiled)
def isentropic_T(total_T, mach, gamma):
    return total_T / (1.0 + (gamma - 1.0) / 2.0 * mach * mach)
#@jit(nopython=isCompiled)
def isentropic_p(total_p, mach, gamma):
    return total_p * (1.0 + (gamma - 1.0) / 2.0 * mach*mach)**(-gamma/(gamma-1.0))
#@jit(nopython=isCompiled)
def evaluate_e(rho, T, u, Cv ):
    return rho * (Cv * T + 0.5 * u*u)
#@jit(nopython=isCompiled)
def evaluate_p(W, gamma):
    return (gamma - 1.0) * (W[2,:] - (W[1,:]**2/W[0,:])/2)
#@jit(nopython=isCompiled)
def evaluate_p1(W1, gamma):
    return (gamma - 1.0) * (W1[2] - (W1[1]**2/W1[0])/2)
#@jit(nopython=isCompiled)
def evaluate_primitive_from_state(W, gamma):
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    p = evaluate_p(W, gamma)
    return rho, u, p
#@jit(nopython=isCompiled)
def evaluate_all(W, gamma):
    rho, u, p = evaluate_primitive_from_state(W, gamma)
    c = evaluate_c(p, rho, gamma)
    mach = u/c
    return rho, u, p, c, mach

def evaluate_source_state(p, area):
    Q = np.zeros([3, p.size])
    Q[1,:] = p * np.diff(area)
    return Q
def evaluate_convective_state(W, gamma):
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    p = evaluate_p(W, gamma)
    e = W[2,:]

    F = np.empty_like(W)
    F[0,:] = rho
    F[1,:] = rho*u*u + p
    F[2,:] = ( e + p ) * u;
    return F

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

    fluxes = np.empty([3, rho.size+1])
    fluxes[:,1:-1] = 0.5*((F[:,:-1] + F[:,1:]) - scalar_eps * lamb * (W[:,1:] - W[:,:-1]))
    return fluxes

#@jit(nopython=isCompiled)
def BC_inlet_residual(W_g, W_d, dt, dx, inlet_total_p, inlet_total_T, Cv, gamma, a2):
    R = Cv*(gamma-1.0)
    #r_g = np.copy(W_g[0])
    r_g = W_g[0]
    u_g = W_g[1] / r_g
    p_g = evaluate_p1(W_g, gamma)
    c_g = evaluate_c(p_g, r_g, gamma)

    r_d = W_d[0]
    u_d = W_d[1] / W_d[0]
    p_d = evaluate_p1(W_d, gamma)
    c_d = evaluate_c(p_d, r_d, gamma)

    dW_g = np.zeros(3)
    if u_g < c_g:
        g = gamma
        gm1 = g - 1.0; gp1 = g + 1.0
        dpdu = inlet_total_p * (g / gm1) \
               * (1.0 - (gm1 / gp1) * u_g * u_g / a2)**(1.0 / gm1) \
               * (-2.0 * (gm1 / gp1) * u_g / a2);

        dtdx = dt / dx
        eigenvalue = ((u_g-c_g + u_d-c_d) / 2.0) * dtdx;
        dpdx = p_d-p_g
        dudx = u_d-u_g

        du = -eigenvalue * (dpdx - r_g * c_g * dudx) / (dpdu - r_g * c_g);

        u_g_new   = u_g + du
        T_g_new   = inlet_total_T * (1.0 - (gm1 / gp1) * u_g_new * u_g_new / a2);
        p_g_new   = inlet_total_p*(T_g_new / inlet_total_T)**(g / gm1);
        r_g_new   = p_g_new / (R * T_g_new);
        e_g_new   = r_g_new * (Cv * T_g_new + 0.5 * u_g * u_g);

        dW_g[0] = r_g_new           - W_g[0]
        dW_g[1] = r_g_new * u_g_new - W_g[1]
        dW_g[2] = e_g_new           - W_g[2]
    return dW_g

#@jit(nopython=isCompiled)
def BC_outlet_residual(W_g, W_d, dt, dx, Cv, gamma):
    # Returns the update of the ghost vector
    # W_g = Ghost state vector
    # W_d = Domain state vector
    R = Cv*(gamma-1.0)
    #r_g = np.copy(W_g[0]) # Very important else dW_g = 0
    r_g = W_g[0] # Very important else dW_g = 0
    u_g = W_g[1] / r_g
    p_g = evaluate_p1(W_g, gamma)
    c_g = evaluate_c(p_g, r_g, gamma)

    r_d = W_d[0]
    u_d = W_d[1] / W_d[0]
    p_d = evaluate_p1(W_d, gamma)
    c_d = evaluate_c(p_d, r_d, gamma)

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
    T_g_new = p_g_new / (r_g_new * R)
    e_g_new = evaluate_e(r_g_new, T_g_new, u_g_new, Cv)

    dW_g = np.empty_like(W_g)
    dW_g[0] = r_g_new           - W_g[0]
    dW_g[1] = r_g_new * u_g_new - W_g[1]
    dW_g[2] = e_g_new           - W_g[2]
    return dW_g


def evaluate_residual(W, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv):
    # Return the residual of the domain 1:n-1
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    p = evaluate_p(W, gamma)
    c = evaluate_c(p, rho, gamma)

    fluxes = evaluate_fluxes(W, gamma, scalar_eps)
    Q      = evaluate_source_state(p, area)

    residual = fluxes[:,2:-1] * (np.ones((3,1))*area[2:-1]) \
               - fluxes[:,1:-2] * (np.ones((3,1))*area[1:-2]) \
               - Q[:,1:-1]
    return residual
#@jit(nopython=isCompiled)
def update_dt(CFL, dx, W, gamma):
    rho = W[0,:]
    u = W[1,:] / W[0,:]
    p = evaluate_p(W, gamma)
    c = evaluate_c(p, rho, gamma)
    return (CFL * dx) / np.abs(u + c)
def evaluate_dw(W, dx, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2):
    dt = update_dt(CFL, dx, W, gamma)

    dW = np.empty_like(W)

    new_W = np.copy(W)
    for rk_state in range(1,5):
        residual = evaluate_residual(new_W, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv)
        new_W[:,1:-1] = new_W[:,1:-1] - (dt[1:-1] / dx[1:-1]) / (5.0 - rk_state) * residual
    dW = (new_W-W)


    #residual = evaluate_residual(W, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv)
    #dW[:,1:-1] = -(dt[1:-1] / dx[1:-1]) * residual

    dW[:,0] = BC_inlet_residual(W[:,0], W[:,1], dt[0], dx[0], inlet_total_p, inlet_total_T, Cv, gamma, a2)
    dW[:,-1] = BC_outlet_residual(W[:,-1], W[:,-2], dt[-1], dx[-1], Cv, gamma)
    #print(BC_inlet_residual(W[:,0], W[:,1], dt[0], dx[0], inlet_total_p, inlet_total_T, Cv, gamma, a2))
    #print(BC_outlet_residual(W[:,-1], W[:,-2], dt[-1], dx[-1], Cv, gamma))
    #print(dt)
    #print(W+dW)
    #sys.exit()

    return dW

def step_in_time(W, dx, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2):
    dW = evaluate_dw(W, dx, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2)
    W = W + dW
    return W, np.sum(dW[0,:]**2)
def solve_steady(iterations_max, tolerance, it_print, 
                 W, dx, area, 
                 scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2):
    for flow_iteration in range(iterations_max):
        W, normR = step_in_time(W, dx, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2)
        #normR = np.sum((residual[0,:])**2)
        if normR < tolerance: return W
        if flow_iteration%it_print==0: print("Iterations %d \t Residual1 %e" % (flow_iteration, np.sqrt(normR)))
        if(np.isnan(normR)):
            print("\n\nself.W  \n",W)
            break
    return W
def solver(sim):
    # State     variables: rho, rho*u, e
    # Primitive variables: rho, u, p
    # Auxiliary variables: c, mach

    # Get the simulation constants
    n_elem, iterations_max, it_print, tolerance, CFL, scalar_eps, \
        gamma, R, Cv, inlet_total_T, inlet_total_p, inlet_mach, outlet_p, geom \
        = sim_data.extract_sim(sim)
    # Create a mesh based on given geometry
    x, dx, xh, area, volume = mesh.initialize_mesh(geom, n_elem);

    a2          = 2.0*gamma*Cv*inlet_total_T*((gamma - 1.0) / (gamma + 1.0));
    inlet_p     = isentropic_p(inlet_total_p, inlet_mach, gamma)

    # Used initial temperature to initialize primitive variables
    T = isentropic_T(inlet_total_T, inlet_mach, gamma)

    # Initialize primitive variables
    p = np.linspace(inlet_p, outlet_p, n_elem, endpoint=True)
    rho = evaluate_rho(p, T, R)
    c = evaluate_c(p, rho, gamma)
    u = evaluate_u(c, inlet_mach)
    e = evaluate_e(rho, T, u, Cv)

    W = np.empty([3, n_elem])
    W = np.array([rho, rho*u, evaluate_e(rho, T, u, Cv)])
    F = np.empty_like(W)
    Q = np.empty_like(W)


    fluxes = np.empty([3,n_elem+1])

    dt = np.empty_like(volume)
    residual = np.empty_like(W)
    dW = np.empty_like(W)

    W = solve_steady(iterations_max, tolerance, it_print, \
                     W, dx, area, \
                     scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2)
    return W

    
def main():
    input_data = sim_data.sim

    n_elem, iterations_max, it_print, tolerance, CFL, scalar_eps, \
        gamma, R, Cv, inlet_total_T, inlet_total_p, inlet_mach, outlet_p, geom \
        = sim_data.extract_sim(input_data)

    input_data["geom"] = sim_data.geom_target
    x, dx, xh, area, volume = mesh.initialize_mesh(input_data["geom"], n_elem);
    W = solver(input_data)
    rho, u, p, c, mach = evaluate_all(W, gamma)
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
