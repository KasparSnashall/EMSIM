"""_summary_
A simple program to solve 1D method of moments problems.
From experience method of moments particulalrly with MLFMM
is quick and works well. Hopefully this should be simple to implement
"""
    
import numpy as np
from scipy.sparse.linalg import cg as conjgradsolve
import matplotlib.pyplot as plt


# Have to first set up the matrix to be solved
# THis is particular to how you solve it
# will make a simple feed of a delta gap initially
# Electric field is just V0/dZ and constant
# Also use a solution by fulse functions and point matching

# then need to solve
# this can be done by conjugate gradient
# pretty simple in scipy or BLAS
# scipy.sparse.linalg.cg

def matrix_element_sym(m,n,a):
    # m == n
    # A = 1+4*a**2/dz**2
    # Z = 1/4pi * log (sqrt(A)+1/sqrt(A) -1) - jkdz/4pi
    pass

def matrix_element_asym(m,n,a):
    # m != n
    # Rmq = sqrt((z_m - z_q)**2 + a**2) 
    # Z = SUM from 1 to M of W*((e**-jkR_mq)/4*piu*R_mq)
    pass
    
def build_matrix():
    # need to set up the matrix to be solved
    # this will involve some known e field (b)
    # a known impedance matrix
    # a set of boundary conditions to be applied to the matrix
    # a solution display
    pass
