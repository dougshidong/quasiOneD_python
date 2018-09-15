#! /usr/bin/python3
import sys
import autograd.numpy as np
import matplotlib.pyplot as plt
import autograd

import sim_data
import mesh
import q1d
import myplot as mplt

fig_shape = 1001
fig_p     = 1002

def cost(W, targetP):
    p = q1d.evaluate_p(W, 1.4)
    return np.sqrt(np.sum(p-targetP)**2)
pCostpW_ad = autograd.jacobian(cost, 0)

pCostpW_ad = autograd.jacobian(cost, 0)
pRpW_ad = autograd.jacobian(q1d.evaluate_dw, 0)


#input_data = sim_data.sim
#n_elem, iterations_max, it_print, tolerance, CFL, scalar_eps, \
#    gamma, R, Cv, inlet_total_T, inlet_total_p, inlet_mach, outlet_p, geom \
#    = sim_data.extract_sim(input_data)
#input_data["geom"] = sim_data.geom_target
#x, dx, xh, area, volume = mesh.initialize_mesh(input_data["geom"], n_elem);
#W = q1d.solver(input_data)
#rho, u, p, c, mach = q1d.evaluate_all(W, gamma)
#p_target = p
#area_target = area
#
## Evaluate initial design
#input_data["geom"] = sim_data.geom_initial
#x, dx, xh, area, volume = mesh.initialize_mesh(input_data["geom"], n_elem);
#W = q1d.solver(input_data)
#rho, u, p, c, mach = q1d.evaluate_all(W, gamma)
#p_current = p
#area_current = area
#a2 = 2.0*gamma*Cv*inlet_total_T*((gamma - 1.0) / (gamma + 1.0));
#
#pRpW = pRpW_ad(W, dx, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2)
#print(pRpR)

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
    print(pCostpW_ad(W,p_target))
    a2 = 2.0*gamma*Cv*inlet_total_T*((gamma - 1.0) / (gamma + 1.0));
    #print(f(ad.seed(x)).dvalue)
    #print(pRpW(W, dx, area, scalar_eps, gamma, inlet_total_p, inlet_total_T, Cv, CFL, a2))


    mplt.plot(xh, p_current, fig_id=fig_p, title='Pressure', lab='Current')
    mplt.plot(xh, p_target,  fig_id=fig_p, title='Pressure', lab='Target')

    mplt.plot(x, area_current, fig_id=fig_shape, title='Shape', lab='Current')
    mplt.plot(x, area_target,  fig_id=fig_shape, title='Shape', lab='Target')
    #plt.figure(1)
    #plt.plot(mesh.x, mesh.area,'-o')
    plt.draw()
    plt.pause(1)
    input('Enter to quit')
    plt.close()
if __name__ == "__main__":
    main()
