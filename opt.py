#! /usr/bin/python3
import sys
import autograd.numpy as np
import matplotlib.pyplot as plt
import autograd
import adolc
from adolc import adouble

import sim_data
import mesh
import q1d
import cost
import diff
import myplot as mplt

fig_area  = 1001
fig_p     = 1002

#pCostpW_ad = autograd.jacobian(cost, 0)
#pRpW_ad = autograd.jacobian(q1d.evaluate_dw, 1)


def main():
    sim = sim_data.Simulation_data()

    # Create target pressure
    sim.set_target_geom()
    x, dx, xh, area, volume = mesh.initialize_mesh(sim.geom, sim.n_elem);
    W = q1d.solver(sim)
    rho, u, p, c, mach = q1d.evaluate_all(W, sim.gamma)
    p_target = p
    area_target = area

    # Evaluate initial design
    sim.set_initial_geom()
    x, dx, xh, area, volume = mesh.initialize_mesh(sim.geom, sim.n_elem);
    W = q1d.solver(sim)
    rho, u, p, c, mach = q1d.evaluate_all(W, sim.gamma)
    p_current = p
    area_current = area

    cost_f = cost.inverse_pressure_design(W, p_target)
    print(cost_f)

    derivatives = diff.Differentiation()
    pCostpW = derivatives.pCostpW_adolc(W, p_target)
    print("pCostpW \n", pCostpW)

    pRpW = derivatives.pRpW_adolc(sim, W, area)
    print("Jacobian pRpW \n", pRpW)


    #print(f(ad.seed(x)).dvalue)
    #print("pRpW \n",pRpW_ad(sim, W, dx, area))

    mplt.plot_compare(xh, p_current, p_target, fig_i=fig_p, title_i='Pressure')
    mplt.plot_compare(x, area_current, area_target, fig_i=fig_area, title_i='Area')
    #plt.figure(1)
    #plt.plot(mesh.x, mesh.area,'-o')
    plt.draw()
    plt.pause(1)
    input('Enter to quit')
    plt.close()
if __name__ == "__main__":
    main()
