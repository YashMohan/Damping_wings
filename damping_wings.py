#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:54:03 2022

@author: sharma
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from constants import *
import logging
import h5py
from numba import jit
import statistics
import parameters_file as params
# logger = logging.getLogger("Damping_wings")
#-----------------------------------------------------------------------------

#---------------------------------------------------------------------------
if not os.path.exists(newpath):
    # logger.error("Could not find the directory to load data")
    sys.exit()
#---------------------------------------------------------------------------

#-----------------------------------------------------------------------------
@jit(nopython=True)
def I(x):
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
#Hubble Rate
@jit(nopython=True)
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
@jit(nopython=True)
def optical_depth(den,xh,z,zs):
    
    tau_z = np.zeros(len(z))
    zb = zs
    ze = zs - dl*H(zs)/(c)  # redshift end points of small patches
    tau_gp = (7.16*10**5)*((1+zs)/10)**(3/2)
    
    for i in range(1,n_pixels):
        zb = zs - i*dl*H(zb)/(c)  # Upadting zb and ze from patch to patch
        ze = zb - dl*H(zb)/(c)
        # Calculating the optical depth 
        tau_z = tau_z + np.abs((xh[i]*(den[i]+1))*(tau_gp*R_alpha/np.pi)*(((1+zb)/(1+z))**(3/2))*np.abs((np.abs(I((1+zb)/(1+z)))-np.abs(I((1+ze)/(1+z)))))) # Mesinger
    
    tau_z[0] = 10    # Settting up the lower end, as for blueward side, as there will only be neutral gas and hence the wings will be damped quickly or e^-tau ~ 0
    for k in range(n_pixels-1,0,-1):    # Cut off for blueward side, once tau_z hits a threshold value the damping drops to ~0 blueward to that
        if(tau_z[k]>=5) | (tau_z[k] < 0):
            break

    for l in range(k,0,-1):
        tau_z[l] = 10 
        
    return tau_z
    

#-----------------------------------------------------------------------------

def damping_wings(base,order,Parameters, rank):
    '''
    This code calculates the damping wing optical depth across a given sightline

    Parameters
    ----------
    base : Integer
        Base of the value of halo mass, it is provided to seperate halos and thier data files
    order : Integer
        Order of the value of halo mass, it is provided to seperate halos and thier data files
    rank : Integer, Optional
        Tells the code which parameter file to pick. If no rank is provided then it picks the rank -1, which is the default Parameters file.


    Returns
    -------
    None.

    '''
    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    #Calculating damping wings weighted over density
    
    # Change len_z to twice of n_pixels and make it faster using numba
    
    len_z = n_pixels  # The length of redshift space is same as n_pixels, it allows me to equate tau_z[j] with z while calculating damping wing, hence making the calculation much faster as I'm using the vectors now
    z = np.linspace(Parameters['z']-1.0,Parameters['z']+1.0,num=len_z) #range of z for which the damping wings will be calculated
    delta_z = 1/len_z
    
    tau_avg = np.zeros(len_z)       # Averaged damping wing optical depth over all the sightlines
    tau_z = np.zeros((N_sightlines,len_z))     # Damping wing optical depth for each sightline
    e_tau_z = np.zeros((N_sightlines,len_z))     # Damping wing optical depth for each sightline
    
    low_quantile = np.zeros(len_z)
    mid_quantile = np.zeros(len_z)
    up_quantile = np.zeros(len_z)
    diff_quantile = np.zeros(len_z)
    var = np.zeros(len_z)
    
    # print("In damping wing function")
    
    # plt.figure(figsize=(5, 6), dpi=150)
    # ax = plt.axes()
    # plt.minorticks_on() 
    # ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
    # ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
    # plt.title(f"Damping wings for ${base}*10^{order} M\odot$")
    
    with h5py.File(f"{newpath}/xh_den_HM_{base}_{order}_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_Rion_sphere_index.h5", 'r') as f:
        Xh = f.get("xh")[:]
        Den = f.get("den")[:]
    
    print('Calculating the optical depth')
    tau_gp = (7.16*10**5)*((1+Parameters['z'])/10)**(3/2)
    
    for j in range(0,N_sightlines):  # Looping over all sightlines
        xh = Xh[j]
        den = Den[j]

        tau_z[j] = optical_depth(den, xh, z, Parameters['z'])
        tau_avg = tau_avg + tau_z[j]/N_sightlines  # Calculating the average damping wing optical depth over all sightlines
        
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
        
    
    # plt.plot(lamda,e_tau_avg,'k--' , label = 'Average value')
    # ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
    # ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
    # plt.xlim(1170,1270)
    # plt.tight_layout()
    # plt.savefig(f"{newpath}/Plots/Damping_wing_Mass_{base}_{order}_rank_{rank}.png")
    # plt.show()
    # plt.close()
    
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


