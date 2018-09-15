#! /usr/bin/python3
import sys
import autograd.numpy as np
import matplotlib.pyplot as plt
import autograd
import sparsegrad.forward as ad

import sim_data
import mesh
import q1d
import myplot as mplt

def cost(W, targetP):
    p = q1d.evaluate_p(W, 1.4)
    return np.sqrt(np.sum(p-targetP)**2)
#pCostpW = autograd.grad(cost, 0)
pCostpW = autograd.jacobian(cost, 0)

pCostpW = autograd.jacobian(cost, 0)
#pRpW = autograd.jacobian(q1d.evaluate_dw, 0)


from casadi import *
input_data = sim_data.sim

n_elem, iterations_max, it_print, tolerance, CFL, scalar_eps, \
    gamma, R, Cv, inlet_total_T, inlet_total_p, inlet_mach, outlet_p, geom \
    = sim_data.extract_sim(input_data)
input_data["geom"] = sim_data.geom_target
x, dx, xh, area, volume = mesh.initialize_mesh(input_data["geom"], n_elem);
W = q1d.solver(input_data)
rho, u, p, c, mach = q1d.evaluate_all(W, gamma)
p_target = p
area_target = area

# Evaluate initial design
input_data["geom"] = sim_data.geom_initial
x, dx, xh, area, volume = mesh.initialize_mesh(input_data["geom"], n_elem);
W = q1d.solver(input_data)
rho, u, p, c, mach = q1d.evaluate_all(W, gamma)
p_current = p
area_current = area
dw = MX.sym("dw")
a2 = 2.0*gamma*Cv*inlet_total_T*((gamma - 1.0) / (gamma + 1.0));
R = q1d.evaluate_dw(dw, dx, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2)
pRpW = gradient(R,dw)
f_pRpW = Function('f',[x],[grad_y])
print(f_pRpW(W))

def main():
    input_data = sim_data.sim

    n_elem, iterations_max, it_print, tolerance, CFL, scalar_eps, \
        gamma, R, Cv, inlet_total_T, inlet_total_p, inlet_mach, outlet_p, geom \
        = sim_data.extract_sim(input_data)

    # Create target pressure
    input_data["geom"] = sim_data.geom_target
    x, dx, xh, area, volume = mesh.initialize_mesh(input_data["geom"], n_elem);
    W = q1d.solver(input_data)
    rho, u, p, c, mach = q1d.evaluate_all(W, gamma)
    p_target = p
    area_target = area

    # Evaluate initial design
    input_data["geom"] = sim_data.geom_initial
    x, dx, xh, area, volume = mesh.initialize_mesh(input_data["geom"], n_elem);
    W = q1d.solver(input_data)
    rho, u, p, c, mach = q1d.evaluate_all(W, gamma)
    p_current = p
    area_current = area

    print(cost(W, p_target))
    print(pCostpW(W,p_target))
    a2 = 2.0*gamma*Cv*inlet_total_T*((gamma - 1.0) / (gamma + 1.0));
    #print(f(ad.seed(x)).dvalue)
    #print(pRpW(W, dx, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2))


    mplt.plot(xh, p_current, fig_id=1, title='Pressure', lab='Current')
    mplt.plot(xh, p_target, fig_id=1, title='Pressure', lab='Target')
    #plt.figure(1)
    #plt.plot(mesh.x, mesh.area,'-o')
    plt.draw()
    plt.pause(1)
    input('Enter to quit')
    plt.close()
if __name__ == "__main__":
    main()
