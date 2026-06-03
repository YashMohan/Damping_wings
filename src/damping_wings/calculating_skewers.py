#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:44:34 2022

@author: sharma

cleaned up versions of the code
"""

#-----------------------------------------------------------------------------
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
import random
import h5py
import os
import sys
from numba import jit
import pickle
import py21cmfast as p21c
from .config.constants import n_pixels, dl, N_sightlines, newpath, H0, Omega_b, Omega_k, Omega_lambda, Omega_m, Nion, Conversion_amu_Mpc, L_Box, HII_DIM, DIM, txt_files, seed
#-----------------------------------------------------------------------------
# Seed
np.random.seed(1000)
random.seed(1000)

if not os.path.exists(newpath):
    print("Could not find the directory to load data")
    sys.exit()
    
class SimParams(TypedDict):
    x_hi: float
    m_min: float
    t_q: float
    m_qso: float
    redshift: float

#-----------------------------------------------------------------------------

dr = np.arange(0,n_pixels,dl)    # Distance travelled across any vector

#-----------------------------------------------------------------------------
#   Point on the surface of unit sphere

# @jit(nopython=True)
def sample_spherical(npoints: int, ndim: int= 3) -> NDArray[np.float64]:
    '''
    This function gives n_points on the surface of n_dimensional sphere

    Parameters
    ----------
    npoints : Int
        Number of points on the surface of the sphere
    ndim : Int, optional
        Dimension of the sphere. The default is 3.

    Returns
    -------
    vec : array
        returns the coordinate vector of the point(s).

    '''
    vec: NDArray[np.float64]

    vec = np.random.randn(ndim, npoints)
    vec = vec/np.linalg.norm(vec, axis = 0)
    return vec

#-----------------------------------------------------------------------------
#Hubble Rate
# @jit(nopython=True)
def H(z: float) -> float:
    '''
    This function calculates the value of Hubble constant for the given constants of universe at a given redshift 'z'

    Parameters
    ----------
    z : float 
        Redshift at which the Hubble constant neeeds to be evaluated

    Returns
    -------
    H(z) : float
        Returns the Hubble–Lemaître parameter at the given redshift

    '''
    return H0*(Omega_m*(1+z)**3 + Omega_lambda + Omega_k*((1+z)**2))**(1/2)    
#-----------------------------------------------------------------------------
# @jit(nopython=True)
def generate_densities(tq: float, X: float, Y: float, Z: float, interp_ionised_box: NDArray[np.float64], interp_density_field: NDArray[np.float64], Parameters: SimParams) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    #-----------------------------------------------------------------------------

    len_z: int = n_pixels
    z_red: NDArray[np.float64] = np.linspace(Parameters['z'],Parameters['z']-1.0,num=len_z)  # redshift space over which we will calculate our damping wings, it has the same size as the number of pixels 
    tol: float = 0.5    # Tolerence value for ionized sphere around a halo
    Rion: float = 0.0   # Radius of the sphere
    xh: NDArray[np.float64] = np.zeros(n_pixels) # Calculating the x_h along (X,Y,Z)
    den: NDArray[np.float64] = np.zeros(n_pixels) # Calculating the density along (X,Y,Z)
    sum: NDArray[np.float64] = np.zeros(n_pixels)   # Calculating the total number of ionized gas within a Range of radius r carved by quasa
    r: NDArray[np.float64] = np.zeros(n_pixels)    # Calculating the length of the skewer
    # sphere_index: int = 0
    xh: NDArray[np.float64]     # Interpolated values of xH along the skewer
    den: NDArray[np.float64]    # Interpolated values of density along the skewer

    #-----------------------------------------------------------------------------
    # Need to get the quasar lifetime drawn from a distribution, rather than just a single number
    # The distribution will be lognormal with tq as mean value and std deviation of 0.5
    # dist = lognorm.pdf(total,mean,stddev)     
    if tq == 0:    # If the quasar is off
        for j in range (0,n_pixels):
            xh[j] = interp_ionised_box([X[j],Y[j],Z[j]])  
            den[j] = interp_density_field([X[j],Y[j],Z[j]])         
            
        return xh,den, Rion
            
    # If the quasar is on then it will carve up some Range of ionized gas around the halo with the 'radius' = Rion. Note: It is not exacatly a sphere
    

    for k in range(0,n_pixels-1):
        del_r = -(z_red[k+1] - z_red[k])*c/(H(z_red[k])*(1+z_red[k]))
        r[k+1] = r[k] + del_r
        XH = interp_ionised_box([X[k],Y[k],Z[k]])
        DEN = interp_density_field([X[k],Y[k],Z[k]])
        
        rho_c = (3*H(z_red[k])**2)*Conversion_amu_Mpc/(8*np.pi*G)
        n_HI = Omega_b*rho_c*XH*(DEN+1)
        sum[k+1] = sum[k] + n_HI*4*np.pi*r[k+1]*r[k+1]*del_r  

    for l in range(n_pixels-1,0,-1):        
        if(np.abs(sum[l] - Nion*tq)/(Nion*tq)<=tol):
            Rion = r[l]  # Radius of the ionized sphere
            break
    for j in range (0,n_pixels):
        if(dr[j]<=Rion*(1+z_red[j])):    # add (1+z) for co-moving space
            xh[j] = 0.0  # Within Rion, all the gas is ionized
            # sphere_index = j
        else:
            xh[j] = interp_ionised_box([X[j],Y[j],Z[j]])   
        den[j] = interp_density_field([X[j],Y[j],Z[j]])
    
    return xh, den, Rion
    

#-----------------------------------------------------------------------------
def calculate_skewers(base_halo_mass: int, o_halo_mass: int, new_halo_coords: NDArray[np.float64], ionised_box: p21c.IonizedBox, density_field: p21c.PerturbedField ,Parameters: SimParams, rank: int) -> None:    
    '''
    Description:
        This code calculates the neutral fraction weighted over density for a halo along some random sightlines for a given ionized box

    Parameters
    ----------
    base_halo_mass : Integer
        Base of the value of halo mass, it is provided to seperate halos and thier data files
    o_halo_mass : Integer
        Order of the value of halo mass, it is provided to seperate halos and thier data files
    new_halo_coords : Array
        x,y,z coordinates of the halos in the given mass bin
    ionised_box : Array
        Neutral fraction at each pixels in the box
    density_field : Array
        Density of matter at each pixel in the box
    rank : Integer, Optional
        Tells the code which parameter file to pick. If no rank is provided then it picks the rank -1, which is the default Parameters file.


    Returns
    -------
    None.

    '''
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Storing x,y and z coords of the halos seperately
    halo_x_coord: NDArray[np.float64]   # Coords of halos, x coordinate of all halos
    halo_y_coord: NDArray[np.float64]   # Coords of halos, y coordinate of all halos
    halo_z_coord: NDArray[np.float64]   # Coords of halos, z coordinate of all halos
    xx: NDArray[np.float64]   # Range of x coordinates, used for interpolation of ionized and density fields
    yy: NDArray[np.float64]   # Range of y coordinates, used for interpolation of ionized and density fields
    zz: NDArray[np.float64]   # Range of z coordinates, used for interpolation of ionized and density fields
    interp_ionised_box: NDArray[np.float64]  # Interpolated inonisation field
    interp_density_field: NDArray[np.float64]   # Interpolated density field
    X: NDArray[np.float64]  # X coordinates of all the skewers
    Y: NDArray[np.float64]  # Y coordinates of all the skewers 
    Z: NDArray[np.float64]  # Z coordinates of all the skewers
    Random_halo: NDArray[np.float64]   # A set of random halos selected from the total list of halos
    Rion: NDArray[np.float64]   # The radius of the bubble ionized by the halo
    sphere_index: NDArray[np.float64]   # The pixel of the radius along the length of skewer
    xi: float # x coordinate of random direction vector
    yi: float # y coordinate of random direction vector
    zi: float # z coordinate of random direction vector
    mu: float # Mean of the lognormal distribution of quasar lifetime
    sigma: float # Stadndard deviation of the lognormal distribution of quasar lifetime
    tq_final: float # Updated quasar lifetime with mu and sigma, to get the quasar lifetime drawn from a distribution, rather than just a single number. In secs
    xh: NDArray[np.float64] # Neutral fraction along the length of skewer
    den: NDArray[np.float64] # Density along the lenght of skewer
   
    halo_x_coord = new_halo_coords[:,0]*L_Box/HII_DIM
    halo_y_coord = new_halo_coords[:,1]*L_Box/HII_DIM
    halo_z_coord = new_halo_coords[:,2]*L_Box/HII_DIM
    
    
    xx = np.linspace(0,HII_DIM-1,HII_DIM)
    yy = np.linspace(0,HII_DIM-1,HII_DIM)
    zz = np.linspace(0,HII_DIM-1,HII_DIM)
    
    interp_ionised_box = RegularGridInterpolator((xx,yy,zz), ionised_box)
    interp_density_field = RegularGridInterpolator((xx,yy,zz), density_field)
    
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    #Calculting the skewers
    
    print(f"\nCalculating skewers for base: {base_halo_mass}, order: {o_halo_mass}")
    
    # Storing the coordinates along the sightline for each halo
    
    X = [[]*n_pixels]*N_sightlines
    Y = [[]*n_pixels]*N_sightlines
    Z = [[]*n_pixels]*N_sightlines   
    xh = [[]*n_pixels]*N_sightlines 
    den = [[]*n_pixels]*N_sightlines 
           
    Random_halo = np.zeros(N_sightlines, dtype=int) 
    
    Rion = np.zeros(N_sightlines)
    sphere_index = np.zeros(N_sightlines)
    
    for i in range(0,N_sightlines):
        Random_halo[i] = random.randrange(len(halo_x_coord))    # Picks up a random halo from the given range of halos

        
    for i in range(0,N_sightlines):   # Loop over all sightlines
        
        xi, yi, zi = sample_spherical(1)

        tq = Parameters['tq']/86400/365.25 # In years 
        mu, sigma = np.log10(tq), 0.8
        log_tq = np.random.normal(mu, sigma, 1)
        tq_final = 10**(log_tq)*86400*365.25
        
        X[i] = halo_x_coord[Random_halo[i]] + dr*xi
        Y[i] = halo_y_coord[Random_halo[i]] + dr*yi
        Z[i] = halo_z_coord[Random_halo[i]] + dr*zi
        
        
        #Enabling the periodic boundary condition
        X[i] = X[i] - (HII_DIM-1)*(np.floor(X[i])//(HII_DIM-1))
        Y[i] = Y[i] - (HII_DIM-1)*(np.floor(Y[i])//(HII_DIM-1))
        Z[i] = Z[i] - (HII_DIM-1)*(np.floor(Z[i])//(HII_DIM-1))
        
        xh[i], den[i], Rion[i] = generate_densities(tq_final, X[i], Y[i], Z[i], interp_ionised_box, interp_density_field, Parameters)
        
        
        
    with h5py.File(f"{newpath}/xh_den_HM_{base_halo_mass}_{o_halo_mass}_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.h5", 'w') as f:    
        f.create_dataset("xh", data = xh)
        f.create_dataset("den", data = den)
        f.create_dataset("Rion", data = Rion)
        # f.create_dataset("sphere_index", data = sphere_index)
  
#--------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------
    

if __name__ == '__main__':
    
    rank = 0
    
    halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
    halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
    ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
    density_field = pickle.load( open(f"{newpath}/Density_field_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
    
    base = []
    order = []
    num_halos = []
    mass_halos = []
    # file = open(f"{txt_files}/Halos_for_skewers_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt",'r')
    file = open(f"{txt_files}/Halos_for_skewers_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.txt",'r')
    for l in file.readlines():
        b, o, n, m = l.strip().split(" ")
        base.append(int(b))
        order.append(int(o))
        num_halos.append(n)
        mass_halos.append(m)
        
    base = np.array(base)
    order = np.array(order)
    num_halos = np.array(num_halos)
    mass_halos = np.array(mass_halos)
    
    
    