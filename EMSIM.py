
"""
This program is written as an exersise to try and create a EM simulation program w
with multiple different simulation techniques included.

Ideally the program will use a GUI for modelling this might end up being some other open source code. 

Based on the book Computational Electromagnetics by Rylander, Bondeson, and Ingelstorm
"""

import numpy as np 
import matplotlib.pyplot as plt	
#from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph as pg
#import pyqtgraph.opengl as gl
import xarray as xr
#import dask
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *  
from pathlib import Path
#import freecad as FreeCAD
#import FreeCADGui as Gui

global eps0 
eps0 = 8.8541878e-12  # Permittivity of vacuum
global mu0
mu0  = 4e-7 * np.pi    # Permeability of vacuum
global c0
c0   = 299792458       # Speed of light in vacuum

class FEM:
	"""Conatiner for the FEM simulations"""
	def __init__(self, arg):
		super().__init__(arg)

class FDTD:
	""" Container for the FDTD method"""

	def __init__(self):
		super().__init__()
		self.frequency = 5e9

	def solve(self):
		#activates the solver
		pass

	def port(self):
		#will create a port definition
		pass

	def boundary_condiitions(self):
		#sets the boundary conditions
		pass

	def solve2D(self):

		# Hx = mHx1.*Hx +mHx2.*CEx + mHx3.*ICEx matlab implementation
		
		Hx = np.multiply(mHx1, Hx) + np.multiply(mHx2, CEx) + np.multiply(mHx3, ICEx)
		pass

	def FDTD_grid_2D(self):
		"""Will create an FDTD grid"""

		wavelength = self.frequency

		#Nx, Ny, Nx should be defined as the cavity size divided by the wavelength

		Nx = 25 # this will be controlled automatically by a wavelength parameter
		Ny = 20
		Nz = 15

		Lx = 0.05 # cavity size in meters, of course this would be an input eventually
		Ly = 0.04
		Lz = 0.03

		Cx = Nx/Lx # normaluised box size
		Cy = Ny/Ly
		Cz = Nz/Lz

		Number_Timsteps = 8192
		Timestep = 1/(c0*np.linalg.norm([Cx, Cy, Cz]))

		#grid = np.zeros([nx,ny]) # even grid

		Ex = np.zeros([Nx, Ny+1, Nz+1 ]) 
		Ey = np.zeros([Nx+1, Ny, Nz+1 ]) 
		Ez = np.zeros([Nx+1, Ny+1, Nz]) 

		Hx = np.zeros([Nx+1, Ny, Nz]) 
		Hy = np.zeros([Nx, Ny+1, Nz]) 
		Hz = np.zeros([Nx, Ny, Nz+1]) 

		Et = np.zeros([Number_Timsteps])  

		Ex[:,1:Ny,1:Nz] = np.random.rand(Nx, Ny-1, Nz-1) - 0.5
		Ey[1:Nx,:,1:Nz] = np.random.rand(Nx-1, Ny, Nz-1) - 0.5
		Ez[1:Nx,1:Ny,:] = np.random.rand(Nx-1, Ny-1, Nz) - 0.5

		Et_plane = []
		#print(Ex)
		for n in range(Number_Timsteps-1):
			Hx = Hx + (Timestep/mu0)*(np.diff(Ey,1,2)*Cz-np.diff(Ez,1,1)*Cy) # dask diff
			Hy = Hy + (Timestep/mu0)*(np.diff(Ez,1,0)*Cx-np.diff(Ex,1,2)*Cy)
			Hz = Hz + (Timestep/mu0)*(np.diff(Ex,1,1)*Cy-np.diff(Ey,1,0)*Cx)

			Ex[:,1:Ny,1:Nz] = Ex[:,1:Ny,1:Nz] + (Timestep/eps0)*(np.diff(Hz[:,:,1:Nz],1,1)*Cy-np.diff(Hy[:,1:Ny,:],1,2)*Cz)
			Ey[1:Nx,:,1:Nz] = Ey[1:Nx,:,1:Nz] + (Timestep/eps0)*(np.diff(Hx[1:Nx,:,:],1,2)*Cz-np.diff(Hz[:,:,1:Nz],1,0)*Cx)
			Ez[1:Nx,1:Ny,:] = Ez[1:Nx,1:Ny,:] + (Timestep/eps0)*(np.diff(Hy[:,1:Ny,:],1,0)*Cx-np.diff(Hx[1:Nx,:,:],1,1)*Cy)

			Et[n] = Ex[4,4,4] + Ey[4,4,4] + Ez[4,4,4] # sampling to get the eigen modes could keep this per plane, then FFT for each field
			
			E_sum = np.zeros(np.shape(Ex[:-1,:-1,4]))
			#for i, eval in np.ndenumerate(E_sum):
			#		E_sum[i] = Ex[i] + Ey[i] + Ez[i]

			Et_plane.append(Ez[:,:,4])# + Ey[:,:,4] + Ez[:,:,4])


		def analytical(m,n,p):
			return c0/2*((m/Lx)**2+(n/Ly)**2+(p/Lz)**2)**(0.5)

		#print(np.array(Et).shape)
		print(Et_plane[0])
		signal = np.fft.fft(Et)
		n = signal.size
		freq = np.fft.fftfreq(n, d=Timestep)
		plt.plot(freq,np.abs(signal))
		plt.vlines(analytical(1,1,2),0,100, colors = "orange", linestyles="--")
		plt.vlines(analytical(1,1,3),0,100, colors = "orange", linestyles="--")

		plt.vlines(analytical(2,1,1),0,100, colors = "orange", linestyles="--")
		plt.vlines(analytical(3,1,1),0,100, colors = "orange", linestyles="--")
		plt.vlines(analytical(0,0,1),0,100, colors = "orange", linestyles="--")
		plt.vlines(analytical(1,0,0),0,100, colors = "orange", linestyles="--")
		plt.vlines(analytical(0,1,0),0,100, colors = "orange", linestyles="--")
		plt.show()


	def make_PML_layer(self, grid):
		""" This will make a pml boundary layer,
		This is more difficult that simple electric boundaries
		

		PML layer is defined as having a sigma of 0.5*(epsilon0/(2*dt))*(nx/NMPL)

		it is done on a 2 times grid

		need a proper reference for this

		"""
		
		pass

	def check_time_stability(self):
		pass

if __name__ == "__main__":
	mysolver = FDTD()
	mygrid = mysolver.FDTD_grid_2D()



