#! /usr/bin/python3
import numpy as np

def inverse_pressure_design(W, p_target):
    p = (1.4 - 1.0) * (W[2,:] - (W[1,:]**2/W[0,:])/2)
    return np.sqrt(np.sum(p-p_target)**2)
