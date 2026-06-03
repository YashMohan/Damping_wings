#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:54:03 2022

@author: sharma
"""
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import os
import sys
import logging
import h5py
from numba import jit
import statistics
import py21cmfast as p21c
from .config.constants import n_pixels, dl, N_sightlines, newpath, H0, Omega_b, Omega_k, Omega_lambda, Omega_m, Nion, Conversion_amu_Mpc, L_Box, HII_DIM, DIM, txt_files, R_alpha, c
import .config.parameters_file as params
#-----------------------------------------------------------------------------

#---------------------------------------------------------------------------
if not os.path.exists(newpath):
    sys.exit()
#---------------------------------------------------------------------------

class SimParams(TypedDict):
    x_hi: float
    m_min: float
    t_q: float
    m_qso: float
    redshift: float
#-----------------------------------------------------------------------------

@jit(nopython=True)
def I(x: float) -> float:
    '''
    Integration function from Mesinger:2008

    Parameters
    ----------
    x : Float
        Ratio of ((1+zb(ze))/(1+z))

    Returns
    -------
    Float
        Returns the value of the integration function from the Mesinger paper

    '''

    return ((x**(1/2))/(1-x) + 2*np.log(np.abs((1-x**(1/2))/(1+x**(1/2)))))
 
#-----------------------------------------------------------------------------

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

@jit(nopython=True)
def optical_depth(den: NDArray[np.float64], xh: NDArray[np.float64], z: NDArray[np.float64], zs: float) -> NDArray[np.float64]:
    '''
    This function returns the Lyman alpha optical depth per pixel along the line of sight of the skewer
    Parameters
    ----------
        den : NDArray[np.float64]
            The density field along the length of skewer sightline
        xh : NDArray[np.float64]
            The neutral fraction along the length of skewer sightline
        z : NDArray[np.float64]
            The redshift range along the length of skewer sightline
        zs : float
            The redshift of the source quasar

    Returns
    -------
        tau_z : NDArray[np.float64]
            Lyman alpha optical depth per pixel along the length of skewer sightline
    '''
    tau_z:  NDArray[np.float64] # Lyman alpha optical depth
    zb: float   # Beginning redshift
    ze: float   # End redshift
    
    
    tau_z = np.zeros(len(z))
    zb = zs
    ze = zs - dl*H(zs)/(c)  # redshift end points of small patches
    tau_gp = (7.16*10**5)*((1+zs)/10)**(3/2)
    
    for i in range(1,n_pixels):
        zb = zs - i*dl*H(zb)/(c)  # Updating zb and ze from patch to patch
        ze = zb - dl*H(zb)/(c)
        # Calculating the optical depth 
        tau_z = tau_z + np.abs((xh[i]*(den[i]+1))*(tau_gp*R_alpha/np.pi)*(((1+zb)/(1+z))**(3/2))*np.abs((np.abs(I((1+zb)/(1+z)))-np.abs(I((1+ze)/(1+z)))))) # Mesinger
    
    tau_z[0] = 10    # Setting up the lower end, as for blueward side, as there will only be neutral gas and hence the wings will be damped quickly or e^-tau ~ 0
    for k in range(n_pixels-1,0,-1):    # Cut off for blueward side, once tau_z hits a threshold value the damping drops to ~0 blueward to that
        if(tau_z[k]>=5) | (tau_z[k] < 0):
            break

    for l in range(k,0,-1):
        tau_z[l] = 10 
        
    return tau_z
    

#-----------------------------------------------------------------------------

def damping_wings(base: float, order: float, Parameters: SimParams, rank: int = 0) -> None:
    '''
    This code calculates the damping wing optical depth across a given sightline

    Parameters
    ----------
    base : Integer
        Base of the value of halo mass, it is provided to separate halos and their data files
    order : Integer
        Order of the value of halo mass, it is provided to separate halos and their data files
    rank : Integer, Optional
        Tells the code which parameter file to pick. If no rank is provided then it picks the rank -1, which is the default Parameters file.

    Returns
    -------
    None.

    '''
    
    len_z: NDArray[np.float64]  # Pixel length of redshift array
    z: NDArray[np.float64]  # Redshift along the skewer sightline 
    lamda: NDArray[np.float64]  # Wavelength range of with respect to the observer
    tau_gp: NDArray[np.float64] # Gunn-Peterson optical depth
    tau_avg: NDArray[np.float64]  # Average Lyman alpha optical depth over all sightlines
    tau_z: NDArray[np.float64]  # Lyman alpha optical depth for each sightline
    e_tau_z: NDArray[np.float64]  # Transmission flux
    low_quantile: NDArray[np.float64]   # 68% quantile lower limit for the ensemble of Lyman alpha damping wing profiles
    mid_quantile: NDArray[np.float64]   # Median of the ensemble of Lyman alpha damping wing profiles
    up_quantile: NDArray[np.float64]    # 68% quantile upper limit for the ensemble of Lyman alpha damping wing profiles
    diff_quantile: NDArray[np.float64]  # Width of the 68% quantile
    var: NDArray[np.float64]    # Variance of the Lyman alpha damping wing profiles distribution
    sd: NDArray[np.float64]  # Standard deviation of the Lyman alpha damping wing profiles distribution
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------    
    # Change len_z to twice of n_pixels and make it faster using numba
    
    len_z = n_pixels  # The length of redshift space is same as n_pixels, it allows me to equate tau_z[j] with z while calculating damping wing, hence making the calculation much faster as I'm using the vectors now
    z = np.linspace(Parameters['z']-1.0,Parameters['z']+1.0,num=len_z) #range of z for which the damping wings will be calculated
    
    tau_avg = np.zeros(len_z)       # Averaged damping wing optical depth over all the sightlines
    tau_z = np.zeros((N_sightlines,len_z))     # Damping wing optical depth for each sightline
    e_tau_z = np.zeros((N_sightlines,len_z))     # Damping wing optical depth for each sightline
    
    low_quantile = np.zeros(len_z)
    mid_quantile = np.zeros(len_z)
    up_quantile = np.zeros(len_z)
    diff_quantile = np.zeros(len_z)
    var = np.zeros(len_z)
    
    # print("In damping wing function")    
    with h5py.File(f"{newpath}/xh_den_HM_{base}_{order}_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_Rion_sphere_index.h5", 'r') as f:
        Xh = f.get("xh")[:]
        Den = f.get("den")[:]
    
    print('Calculating the optical depth')
    tau_gp = (7.16*10**5)*((1+Parameters['z'])/10)**(3/2)
    
    for j in range(0,N_sightlines):  # Looping over all sightlines
        xh = Xh[j]
        den = Den[j]

        tau_z[j] = optical_depth(den, xh, z, Parameters['z'])
        tau_avg = tau_avg + tau_z[j]/N_sightlines 
        
        e_tau_z[j] = np.exp(-tau_z[j])
        lamda = (1+z)*1215.67/(1+Parameters['z'])   # Observed wavelength
        # plt.plot(lamda,e_tau_z[j])
        
    e_tau_avg = np.exp(-tau_avg)
    
    
    with h5py.File(f"{newpath}/skewers_HM_{base}_{order}_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_Rion.h5", 'w') as f:    
        f.create_dataset("lambda", data = lamda)
        f.create_dataset("tau", data = tau_z)
        f.create_dataset("tau_avg", data = tau_avg)
        f.create_dataset("e_tau", data = e_tau_z)
        f.create_dataset("e_tau_avg", data = e_tau_avg)
        

    
    print('Calculating the quantiles')
    for i in range(0,len_z):
        low_quantile[i] = np.quantile(e_tau_z[:,i],0.16)
        mid_quantile[i] = np.quantile(e_tau_z[:,i],0.50)
        up_quantile[i] = np.quantile(e_tau_z[:,i],0.84)
        diff_quantile[i] = up_quantile[i] - low_quantile[i]
        var[i] = statistics.variance(e_tau_z[:,i])
    
    sd = np.sqrt(var)    
    with h5py.File(f"{newpath}/quantile_data_{base}_{order}_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_Rion.h5", 'w') as f:  
        
        f.create_dataset("low_quantile", data = low_quantile)
        f.create_dataset("mid_quantile", data = mid_quantile)
        f.create_dataset("up_quantile", data = up_quantile)
        f.create_dataset("diff_quantile", data = diff_quantile)
        
    with h5py.File(f"{newpath}/statistics_{base}_{order}_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_Rion.h5", 'w') as f:  
        
        f.create_dataset("variance", data = var)
        f.create_dataset("standard_deviation", data = sd)
        


        
if __name__ == '__main__':
    
    Parameters = params.Parameters
    
    # damping_wings(4,11,Parameters, rank=0)
    
    r = [1,2]
    for rank in r:
        damping_wings(4, 11, Parameters, rank)


