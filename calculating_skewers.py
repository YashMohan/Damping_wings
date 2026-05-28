#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:44:34 2022

@author: sharma

cleaned up versions of the code
"""

#-----------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import random
import h5py
import os
import sys
from constants import *
import logging
from numba import jit
import pickle
import scipy.stats import lognorm

# logger = logging.getLogger("Damping_wings")
#-----------------------------------------------------------------------------
# Seed

np.random.seed(1000)
random.seed(1000)

if not os.path.exists(newpath):
    #print("Could not find the directory to load data")
    # logger.error("Could not find the directory to load data")
    sys.exit()

#-----------------------------------------------------------------------------

dr = np.arange(0,n_pixels,dl)    # Distance travelled across any vector

#-----------------------------------------------------------------------------
#   Point on the surface of unit sphere

# @jit(nopython=True)
def sample_spherical(npoints, ndim=3):
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
    vec = np.random.randn(ndim, npoints)
    # n = np.sqrt(sum(vec*vec))
    # vec = vec/n
    vec = vec/np.linalg.norm(vec, axis = 0)
    return vec

#-----------------------------------------------------------------------------
#Hubble Rate
# @jit(nopython=True)
def H(z):
    '''
    This function calculates the value of Hubble constant for the given constants of universe at a given redshift 'z'

    Parameters
    ----------
    z : float 
        Redshift at which the Hubble constant neeeds to be evaluated

    '''
    return H0*(Omega_m*(1+z)**3 + Omega_lambda + Omega_k*((1+z)**2))**(1/2)    
#-----------------------------------------------------------------------------
# @jit(nopython=True)
def generate_densities(tq,X,Y,Z, interp_ionised_box, interp_density_field, Parameters):
    #-----------------------------------------------------------------------------

    len_z = n_pixels
    z_red = np.linspace(Parameters['z'],Parameters['z']-1.0,num=len_z)  # redshift space over which we will calculate our damping wings, it has the same size as the number of pixels
    delta_z = 1/(n_pixels)    # differential increment in redshift from pixel to pixel 
    
    tol = 0.5    # Tolerence value for ionized sphere around a halo
    Rion = 0.0   # Radius of the sphere
    xh = np.zeros(n_pixels) # Calculating the x_h along (X,Y,Z)
    den = np.zeros(n_pixels) # Calculating the density along (X,Y,Z)
    Sum = np.zeros(n_pixels)
    r = np.zeros(n_pixels)
    sphere_index = 0

    #-----------------------------------------------------------------------------
    # Need to get the quasar lifetime drawn from a distribution, rather than just a single number
    # The distribution will be lognormal with tq as mean value and std deviation of 0.5
    # dist = lognorm.pdf(total,mean,stddev)     
    if tq == 0:    # If the quasar is off
        for j in range (0,n_pixels):
            xh[j] = interp_ionised_box([X[j],Y[j],Z[j]])  # Interpolating the values of xH and density all along the sightline
            # xh[j] = 0.5
            den[j] = interp_density_field([X[j],Y[j],Z[j]])         
            
        return xh,den, Rion, sphere_index
            
    # If the quasar is on then it will carve up some region of ionized gas around the halo with the 'radius' = Rion. Note: It is not exacatly a sphere
    

    for k in range(0,n_pixels-1):
        del_r = -(z_red[k+1] - z_red[k])*c/(H(z_red[k])*(1+z_red[k]))
        r[k+1] = r[k] + del_r
        XH = interp_ionised_box([X[k],Y[k],Z[k]])
        DEN = interp_density_field([X[k],Y[k],Z[k]])
        
        rho_c = (3*H(z_red[k])**2)*Conversion_amu_Mpc/(8*np.pi*G)
        n_HI = Omega_b*rho_c*XH*(DEN+1)
        Sum[k+1] = Sum[k] + n_HI*4*np.pi*r[k+1]*r[k+1]*del_r  # Calculating the total number of ionized gas within a region of radius r carved by quasar
    
    for l in range(n_pixels-1,0,-1):        
        if(np.abs(Sum[l] - Nion*tq)/(Nion*tq)<=tol):
            Rion = r[l]  # Radius of the ionized sphere
            break
    for j in range (0,n_pixels):
        if(dr[j]<=Rion*(1+z_red[j])):    # add (1+z) for co-moving space
            xh[j] = 0.0  # Within Rion, all the gas is ionized
            sphere_index = j
        else:
            xh[j] = interp_ionised_box([X[j],Y[j],Z[j]])   # Interpolating the values of xH and density all along the sightline
            # xh[j] = 0.5
        den[j] = interp_density_field([X[j],Y[j],Z[j]])
    
    return xh,den, Rion, sphere_index
    

#-----------------------------------------------------------------------------
def calculate_skewers(base_halo_mass,o_halo_mass,new_halo_coords,ionised_box,density_field,Parameters,rank):    
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
   
    halo_x_coord = new_halo_coords[:,0]*L_Box/HII_DIM #Coords of halos, x coordinate of all halos
    halo_y_coord = new_halo_coords[:,1]*L_Box/HII_DIM #y coordinate of all halos
    halo_z_coord = new_halo_coords[:,2]*L_Box/HII_DIM #z coordinate of all halos
    
    
    xx = np.linspace(0,HII_DIM-1,HII_DIM)   #Region of x coordinates
    yy = np.linspace(0,HII_DIM-1,HII_DIM)   #Region of y coordinates 
    zz = np.linspace(0,HII_DIM-1,HII_DIM)   #Region of z coordinates
    # pixels = np.linspace(0,n_pixels-1,n_pixels)     #Pixels range
    
    interp_ionised_box = RegularGridInterpolator((xx,yy,zz), ionised_box)   #Interpolation function to calculate inonisation field
    interp_density_field = RegularGridInterpolator((xx,yy,zz), density_field)   #Interpolation function to calculate density field
    
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
        
        xi, yi, zi = sample_spherical(1)    # random direction vector
        # Need to get the quasar lifetime drawn from a distribution, rather than just a single number
        # The distribution will be lognormal with tq as mean value and std deviation of 0.5
        # dist = lognorm.pdf(total,mean,stddev)

        tq = Parameters['tq']/86400/365.25 # In years 
        mu, sigma = np.log10(tq), 0.8 # mean and standard deviation for lognormal distribution of tq
        log_tq = np.random.normal(mu, sigma, 1)     # Picking a random value of tq out of this distribution
        tq_final = 10**(log_tq)*86400*365.25    # Converting it back into seconds
        
        X[i] = halo_x_coord[Random_halo[i]] + dr*xi #Updating coords along the random direction
        Y[i] = halo_y_coord[Random_halo[i]] + dr*yi #Updating coords along the random direction
        Z[i] = halo_z_coord[Random_halo[i]] + dr*zi #Updating coords along the random direction
        
        
        #Enabling the periodic boundary condition
        X[i] = X[i] - (HII_DIM-1)*(np.floor(X[i])//(HII_DIM-1))
        Y[i] = Y[i] - (HII_DIM-1)*(np.floor(Y[i])//(HII_DIM-1))
        Z[i] = Z[i] - (HII_DIM-1)*(np.floor(Z[i])//(HII_DIM-1))
        
        xh[i], den[i], Rion[i], sphere_index[i] = generate_densities(tq_final, X[i], Y[i], Z[i], interp_ionised_box, interp_density_field, Parameters)
        # print(sphere_index[i])
        
        
        
    with h5py.File(f"{newpath}/xh_den_HM_{base_halo_mass}_{o_halo_mass}_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_Rion_sphere_index.h5", 'w') as f:  
    # with h5py.File(f"{newpath}/xh_den_HM_{base_halo_mass}_{o_halo_mass}_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.h5", 'w') as f:    
        f.create_dataset("xh", data = xh)
        f.create_dataset("den", data = den)
        f.create_dataset("Rion", data = Rion)
        f.create_dataset("sphere_index", data = sphere_index)
  
#--------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------
    

if __name__ == '__main__':
    
    rank = 0
    halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p","rb"))
    halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p","rb"))
    ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p", "rb" ))
    density_field = pickle.load( open(f"{newpath}/Density_field_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p", "rb" ))
    
    # halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
    # halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
    # ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
    # density_field = pickle.load( open(f"{newpath}/Density_field_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
    
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
    
    
    # halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
    # halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
    # ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
    # density_field = pickle.load( open(f"{newpath}/Density_field_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
    
    