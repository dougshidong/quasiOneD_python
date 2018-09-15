#! /usr/bin/python3

sim = {
"n_elem"        : 100,
"iterations_max": 10000,
"it_print"      : 20,
"tolerance"     : 1e-14,

"CFL"           : 0.2,
"scalar_eps"    : 0.3,

"gamma"         : 1.4,
"R"             : 1.0,

"inlet_total_T" : 1.0,
"inlet_total_p" : 1.0,
"inlet_mach"    : 0.85,
"outlet_p"      : 0.82,

"geom"          : [0.10, 0.80, 6.00]
}
sim["tolerance"] = sim["tolerance"]**2
sim["Cv"] = sim["R"] / (sim["gamma"] - 1.0);

geom_initial = [0.07, 0.80, 5.00]
geom_target = [0.05, 1.00, 3.00]

def extract_sim(sim):
    n_elem = sim["n_elem"];
    iterations_max = sim["iterations_max"];
    it_print = sim["it_print"];
    tolerance = sim["tolerance"];
    CFL = sim["CFL"];
    scalar_eps = sim["scalar_eps"];
    gamma = sim["gamma"];
    R = sim["R"];
    Cv = sim["Cv"];
    inlet_total_T = sim["inlet_total_T"];
    inlet_total_p = sim["inlet_total_p"];
    inlet_mach = sim["inlet_mach"];
    outlet_p = sim["outlet_p"];
    geom = sim["geom"];
    return n_elem, iterations_max, it_print, \
           tolerance, CFL, scalar_eps, \
           gamma, R, Cv, \
           inlet_total_T, inlet_total_p, inlet_mach, outlet_p, \
           geom

