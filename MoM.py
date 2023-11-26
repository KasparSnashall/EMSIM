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


def charged_wire_1d():
    
    def build_matrix(n,dx,a=0):
        output = []
        #output.append(0)
        for i in range(1,n):
            xb = (n+1)*dx
            xa = (n+1-1)*dx
            xm = i*dx #this is my current issue?
            
            top = (xb - xm) + np.sqrt((xb-xm)**2-a**2)
            bottom = (xa - xm) + np.sqrt((xa-xm)**2-a**2)
            z = np.log(top / bottom)
            output.append(z)
        
        #output = output[int(n/2):-1] + output[int(n/2):-1] 
        # only need to build the first row then copy N times
        # row = [z1, z2 ... ZN][zn, z1, z2 ...zn-1]
        #output.append(0)
        
        out_matrix = []
        
        for j in range(len(output)):
            out_matrix.append(output)
            output = [output[-1]] + output[0:-1] 

        out_matrix = np.array(out_matrix)
        return out_matrix
        
        
    bm = 1#4*np.pi#*8.854e-12 # normalise
    n = 1501
    dx = 1/(n+1)
    
    a_matrix = build_matrix(n,dx)
    b_matrix = [bm for x in a_matrix]
    
    b_matrix[-1] = 0 # boundary conditions zero field on ends
    b_matrix[0] = 0 # perhaps these are artificial and the issue is in the matirx
    solution = conjgradsolve(a_matrix,b_matrix, tol=1e-9)
    
    plt.plot(abs(solution[0]*8.854e-12*4*np.pi))
    plt.show()

if __name__ == "__main__":
    charged_wire_1d()