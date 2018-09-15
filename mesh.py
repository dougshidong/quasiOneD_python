#! /usr/bin/python3
import numpy as np

def evaluate_volume(area1, area2, dx):
    return 0.5*(area1+area2)*dx
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

