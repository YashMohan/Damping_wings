#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:58:07 2022

@author: sharma

This code calculates the average mass of the pixel of the box to make sure the minimum halo mass is greater than pixel mass
"""

#-----------------------------------------------------------------------------
#Header Files
import os
# We change the default level of the logger so that
# we can see what's happening with caching.
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

import py21cmfast as p21c

# For plotting the cubes, we use the plotting submodule:
from py21cmfast import plotting

# For interacting with the cache
from py21cmfast import cache_tools

import numpy as np

# Provides all the other constants and name of the new path to store the data
from constants import *   

# Parameters
from parameters_file import Parameters    

#-----------------------------------------------------------------------------
'''
# Changing the cache directory for cluster

if not os.path.exists('/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache'):
    os.mkdir('/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache')
p21c.config['direc'] = '/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache'
cache_tools.clear_cache(direc="/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache")
''' 

#-----------------------------------------------------------------------------
#Hubble Rate
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

def Get_me_M_min(initial_conditions):
    '''
    This function calculates the average mass of the pixel of the box for a given set of box parameters

    Returns
    -------
    M_pixels : float
        Returns the average mass of the pixel of the box for a given set of box parameters

    '''

    #-----------------------------------------------------------------------------
    #Setting up the ionized box at the given redshift and initial conditions

    perturbed_field = p21c.perturb_field(
        redshift = Parameters['z'],
        init_boxes = initial_conditions
        )
    
    # Normalized average density of the box
    avg_den = np.mean(perturbed_field.density)
    
    # Critical density
    rho_c = ((3*H(Parameters['z'])**2)/(8*np.pi*G))*(Conversion_kg_Solar_mass/Conversion_m_to_Mpc)
    
    # Average density of baryons
    rho_avg = Omega_m*rho_c*(avg_den+1)  #Msun/Mpc^3

    # Average pixel mass
    M_pixels = rho_avg*(L_Box/(DIM*(1+Parameters['z'])))**3

    return M_pixels


if __name__ == '__main__':
    
    M_p = Get_me_M_min()

    print("{:.2e}".format(M_p))
    

