#! /usr/bin/python3
import numpy as np
import adolc
from adolc import adouble
import q1d
import cost

class Differentiation:
    pCostpW_tag = 1
    pRpW_tag    = 2
    ppRpWpW_tag    = 99
    def __init__(self):
        self.pCostpW_traced = False
        self.pRpW_traced = False
        self.ppRpWpW_traced = False

    def pCostpW_adolc(self, W, p_target):

        tag = self.pCostpW_tag
        if not self.pCostpW_traced:
            aW = adouble(W.flatten(order='F'))
            ap = adouble(p_target)

            adolc.trace_on(tag)
            adolc.independent(aW)
            aW3 = np.reshape(aW, W.shape, order='F')
            acost = cost.inverse_pressure_design(aW3, ap)
            adolc.dependent(acost)
            adolc.trace_off()

        return adolc.gradient(self.pCostpW_tag, W.flatten(order='F'))

    def pRpW_adolc(self, sim, W, area):

        tag = self.pRpW_tag
        if not self.pRpW_traced:
            aW = adouble(W.flatten(order='F'))
            aarea = adouble(area)

            adolc.trace_on(tag)
            adolc.independent(aW)

            aW3 = np.reshape(aW, W.shape, order='F')
            aresidual = q1d.evaluate_residual(sim, aW3, area)

            aresidual.flatten(order='F')
            adolc.dependent(aresidual)
            adolc.trace_off()

        return adolc.jacobian(self.pRpW_tag, W.flatten(order='F'))
    def ppRpWpW_adolc(self, sim, W, area):

        tag = self.ppRpWpW_tag
        if not self.ppRpWpW_traced:
            aW = adouble(W.flatten(order='F'))
            aarea = adouble(area)

            adolc.trace_on(tag)
            adolc.independent(aW)

            aW3 = np.reshape(aW, W.shape, order='F')
            pRpW = self.pRpW_adolc(sim, aW3, area)

            pRpW.flatten(order='F')
            adolc.dependent(pRpW)
            adolc.trace_off()

        return adolc.hessian(self.ppRpWpW_tag, W.flatten(order='F'))
