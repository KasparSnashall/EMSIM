"""_summary_
A simple program to solve 1D method of moments problems.
From experience method of moments particulalrly with MLFMM
is quick and works well. Hopefully this should be simple to implement
"""
    
import numpy as np
from scipy.sparse.linalg import cg as conjgradsolve
import matplotlib.pyplot as plt


# Have to first set up the matrix to be solved
# This is particular to how you solve it
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
    
    # this is assuming a delta gap
    eta = 1 # ?
    k = 1 # wave number
    Zm = 1 # impedance of element m
    b = -(1j/eta)*np.sin(k*Zm)
    # first assume a simple wire with length lambda/2
    # want to calculate the input impedance versus length
    # would like to use a sinusoid basis function but may try others
    pass

def sinusoid_basis(x, k):
    
    # xn-1 < x < xn
    # return np.sin(k*(xn-xn-1))/np.sin(k*(xn-xn-1))
    
    # xn < x < xn+1
    # return np.sin(k*(xn+1-x))/np.sin(k*(xn+1-xn))
    output = []
    for n, i in enumerate(x):
        #if n == 0:
        #    output.append(0)
        #elif n == len(x)-1:
        #    output.append(0)
        #else:
        print(n)
        if n >= len(x)/2:
            fn = np.sin(k*(i-x[n-1]))
        if n < len(x)/2:
            fn = np.sin(k*(x[n+1]-i))
        output.append(fn)
            
    print(output)
    return output

if __name__ == "__main__":
    k = 0.25
    x = np.zeros(25)+1
    print(x)
    
    plt.plot(sinusoid_basis(x,k))
    plt.show()