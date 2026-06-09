#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:44:34 2022

@author: sharma

cleaned up versions of the code
"""

#-----------------------------------------------------------------------------
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
from .utils import H
# Module import for configurable constants — read dynamically at call time
from .config import constants as _constants
# Snapshot imports for physical constants — never change, fine as-is
from .config.constants import H0, Omega_m, Omega_lambda, Omega_k, Omega_b, c, G, h, \
    Nion, R_alpha, Conversion_amu_Mpc, Conversion_kg_Solar_mass, Conversion_m_to_Mpc, \
    dl, n_pixels
#-----------------------------------------------------------------------------
# Seed
np.random.seed(1000)
random.seed(1000)    
#-----------------------------------------------------------------------------
dr = np.arange(0,n_pixels,dl)    # Distance travelled across any vector

#-----------------------------------------------------------------------------
#   Point on the surface of unit sphere
# @jit(nopython=True)
def sample_spherical(npoints: int, ndim: int= 3) -> NDArray[np.float64]:
    '''
    This function gives n_points on the surface of an n-dimensional sphere

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
# @jit(nopython=True)
def generate_densities(
    tq: float,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    interp_ionised_box: RegularGridInterpolator,
    interp_density_field: RegularGridInterpolator,
    Parameters: SimParams
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]: 
    """ This function generates the neutral fraction and matter densities along the length of the skewers. The quasar activity further carves an ionised bubble around
    the halo, hence ionising the nearby region, which is represented by the sphere of radius Rion.

    Args:
        tq (float): quasar lifetime
        X (NDArray[np.float64]): X coordinates of all the skewers
        Y (NDArray[np.float64]): Y coordinates of all the skewers
        Z (NDArray[np.float64]): Z coordinates of all the skewers
        interp_ionised_box (RegularGridInterpolator): interpolated ionisation field
        interp_density_field (RegularGridInterpolator): interpolated density field
        Parameters (SimParams): Simulation parameters

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], float]: neutral fraction and density along the length of the skewer, and the size of the ionised bubble
    """
    len_z: int = n_pixels
    z_red: NDArray[np.float64] = np.linspace(Parameters['z'],Parameters['z']-1.0,num=len_z)  # redshift space over which we will calculate our damping wings, it has the same size as the number of pixels 
    tol: float = 0.5    # Tolerance value for ionized sphere around a halo
    Rion: float = 0.0   # Radius of the sphere
    xh: NDArray[np.float64] = np.zeros(n_pixels) # Calculating the x_h along (X,Y,Z)
    den: NDArray[np.float64] = np.zeros(n_pixels) # Calculating the density along (X,Y,Z)
    n_ion_sum: NDArray[np.float64] = np.zeros(n_pixels)   # Calculating the total number of ionized gas within a Range of radius r carved by the quasar
    r: NDArray[np.float64] = np.zeros(n_pixels)    # Calculating the length of the skewer
    # sphere_index: int = 0
    xh: NDArray[np.float64]     # Interpolated values of xH along the skewer
    den: NDArray[np.float64]    # Interpolated values of density along the skewer
    XH: float
    DEN: float
    n_HI: float                 # Neutral hydrogen density for a given pixel

    #-----------------------------------------------------------------------------
    # Need to get the quasar lifetime drawn from a distribution, rather than just a single number
    # The distribution will be lognormal with tq as the mean value and a standard deviation of 0.5
    # dist = lognorm.pdf(total,mean,stddev)     
    if tq == 0:    # If the quasar is off
        for j in range (0,n_pixels):
            xh[j] = interp_ionised_box([X[j],Y[j],Z[j]])[0]  
            den[j] = interp_density_field([X[j],Y[j],Z[j]])[0]         
            
        return xh, den, Rion
            
    # If the quasar is on, then it will carve up a range of ionized gas around the halo with the 'radius' = Rion. Note: It is not exactly a sphere
    

    for k in range(0,n_pixels-1):
        del_r = -(z_red[k+1] - z_red[k])*c/(H(z_red[k])*(1+z_red[k]))
        r[k+1] = r[k] + del_r
        XH = interp_ionised_box([X[k],Y[k],Z[k]])[0]
        DEN = interp_density_field([X[k],Y[k],Z[k]])[0]
        
        rho_c = (3*H(z_red[k])**2)*Conversion_amu_Mpc/(8*np.pi*G)
        n_HI = Omega_b*rho_c*XH*(DEN+1)
        n_ion_sum[k+1] = n_ion_sum[k] + n_HI*4*np.pi*r[k+1]*r[k+1]*del_r  

    for l in range(n_pixels-1,0,-1):        
        if(np.abs(n_ion_sum[l] - Nion*tq)/(Nion*tq)<=tol):
            Rion = r[l]  # Radius of the ionized sphere
            break
    for j in range (0,n_pixels):
        if(dr[j]<=Rion*(1+z_red[j])):    # add (1+z) for co-moving space
            xh[j] = 0.0  # Within Rion, all the gas is ionized
            # sphere_index = j
        else:
            xh[j] = interp_ionised_box([X[j],Y[j],Z[j]])[0]
        den[j] = interp_density_field([X[j],Y[j],Z[j]])[0]
    
    return xh, den, Rion
    

#-----------------------------------------------------------------------------
def calculate_skewers(base_halo_mass: int, o_halo_mass: int, new_halo_coords: NDArray[np.float64], ionised_box: p21c.IonizedBox, density_field: p21c.PerturbedField ,Parameters: SimParams, rank: int) -> None:    
    '''
    Description:
        This code calculates the neutral fraction weighted over density for a halo along some random sightlines for a given ionized box

    Parameters
    ----------
    base_halo_mass : Integer
        Base of the value of halo mass, it is provided to separate halos and their data files
    o_halo_mass : Integer
        Order of the value of halo mass, it is provided to separate halos and their data files
    new_halo_coords : Array
        x,y,z coordinates of the halos in the given mass bin
    ionised_box : Array
        Neutral fraction at each pixel in the box
    density_field : Array
        Density of matter at each pixel in the box
    rank : Integer, Optional
        Tells the code which parameter file to pick. If no rank is provided, then it picks rank -1, which is the default Parameters file.


    Returns
    -------
    None.

    '''

    if not os.path.exists(_constants.newpath):
        raise FileNotFoundError(
            f"Output directory '{_constants.newpath}' not found. "
            "Call setup_output_dirs() before running the pipeline."
        )
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Storing x,y and z coords of the halos separately
    halo_x_coord: NDArray[np.float64]   # Coords of halos, x coordinate of all halos
    halo_y_coord: NDArray[np.float64]   # Coords of halos, y coordinate of all halos
    halo_z_coord: NDArray[np.float64]   # Coords of halos, z coordinate of all halos
    xx: NDArray[np.float64]   # Range of x coordinates, used for interpolation of ionized and density fields
    yy: NDArray[np.float64]   # Range of y coordinates, used for interpolation of ionized and density fields
    zz: NDArray[np.float64]   # Range of z coordinates, used for interpolation of ionized and density fields
    interp_ionised_box: NDArray[np.float64]  # Interpolated ionisation field
    interp_density_field: NDArray[np.float64]   # Interpolated density field
    X: NDArray[np.float64]  # X coordinates of all the skewers
    Y: NDArray[np.float64]  # Y coordinates of all the skewers 
    Z: NDArray[np.float64]  # Z coordinates of all the skewers
    Random_halo: NDArray[np.float64]   # A set of random halos selected from the total list of halos
    Rion: NDArray[np.float64]   # The radius of the bubble ionized by the halo
    sphere_index: NDArray[np.float64]   # The pixel of the radius along the length of the skewer
    xi: float # x coordinate of random direction vector
    yi: float # y coordinate of random direction vector
    zi: float # z coordinate of random direction vector
    mu: float # Mean of the lognormal distribution of quasar lifetime
    sigma: float # Standard deviation of the lognormal distribution of quasar lifetime
    tq_final: float # Updated quasar lifetime with mu and sigma, to get the quasar lifetime drawn from a distribution, rather than just a single number. In secs
    xh: NDArray[np.float64] # Neutral fraction along the length of skewer
    den: NDArray[np.float64] # Density along the length of skewer
   
    halo_x_coord = new_halo_coords[:,0]*_constants.L_Box/_constants.HII_DIM
    halo_y_coord = new_halo_coords[:,1]*_constants.L_Box/_constants.HII_DIM
    halo_z_coord = new_halo_coords[:,2]*_constants.L_Box/_constants.HII_DIM
    
    
    xx = np.linspace(0,_constants.HII_DIM-1,_constants.HII_DIM)
    yy = np.linspace(0,_constants.HII_DIM-1,_constants.HII_DIM)
    zz = np.linspace(0,_constants.HII_DIM-1,_constants.HII_DIM)
    
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
        X[i] = X[i] - (_constants.HII_DIM-1)*(np.floor(X[i])//(_constants.HII_DIM-1))
        Y[i] = Y[i] - (_constants.HII_DIM-1)*(np.floor(Y[i])//(_constants.HII_DIM-1))
        Z[i] = Z[i] - (_constants.HII_DIM-1)*(np.floor(Z[i])//(_constants.HII_DIM-1))
        
        xh[i], den[i], Rion[i] = generate_densities(tq_final, X[i], Y[i], Z[i], interp_ionised_box, interp_density_field, Parameters)
        
        
        
    with h5py.File(f"{_constants.newpath}/xh_den_HM_{base_halo_mass}_{o_halo_mass}_rank_{rank}_no_halofield_DIM_{_constants.DIM}_HII_{_constants.HII_DIM}_L_{_constants.L_Box}_N_{_constants.N_sightlines}__constants.seed_{_constants.seed}.h5", 'w') as f:    
        f.create_dataset("xh", data = xh)
        f.create_dataset("den", data = den)
        f.create_dataset("Rion", data = Rion)
        # f.create_dataset("sphere_index", data = sphere_index)
  
#--------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------
    

if __name__ == '__main__':
    
    pass
