import numpy as np
#! /usr/bin/python3
class Simulation_data:
    def __init__(self):
        self.n_elem         = 100
        self.iterations_max = 1000
        self.it_print       = 20
        self.tolerance      = 1e-14

        self.CFL            = 0.2
        self.scalar_eps     = 0.3

        self.gamma          = 1.4
        self.R              = 1.0
        self.Cv             = self.R/(self.gamma-1.0)

        self.inlet_total_T  = 1.0
        self.inlet_total_p  = 1.0
        self.inlet_mach     = 0.85
        self.outlet_p       = 0.82

        self.geom_initial  = np.array([0.07, 0.80, 5.00])
        self.geom_target   = np.array([0.05, 1.00, 3.00])
        self.geom          = np.empty(3)

        # Avoid doing a sqrt to compare
        self.tolerance = self.tolerance**2

        self.a2       = 2.0*self.gamma*self.Cv*self.inlet_total_T*((self.gamma - 1.0) / (self.gamma + 1.0))
        self.inlet_p  = self.inlet_total_p * (1.0 + (self.gamma - 1.0) / 2.0 * self.inlet_mach**2)**(-self.gamma/(self.gamma-1.0))

    def set_initial_geom(self):
        self.geom          = self.geom_initial
    def set_target_geom(self):
        self.geom          = self.geom_target

