#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:58:07 2022

@author: sharma

cleaned up version of codes
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

#For new path
import datetime
#----------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------
today = str(datetime.date.today())

'''
if not os.path.exists('/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache'):
    os.mkdir('/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache')
p21c.config['direc'] = '/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache'
cache_tools.clear_cache(direc="/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache")
''' 

#----------------------------------------------------------------------------------------------------------------------------------------------------------
# New path for the code in the laptop

newpath = r'/Users/sharma/work/21cmFast_codes_and_plots/'+today 
if not os.path.exists(newpath):
    os.mkdir(newpath)
#----------------------------------------------------------------------------------------------------------------------------------------------------------
    

#-----------------------------------------------------------------------------
#Parameters

from Parameters_file import Parameters

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
    
#Constants

H0 = 70000.0    # units: m/s/Mpc
Omega_m = 0.3
Omega_lambda = 0.7
Omega_k = 0.0
Omega_b = 0.045
c = 3*10**8 #ms^-1

G = 6.67*10**(-11) # units: m*((m/s)**2)/kg
Conversion_m_to_Mpc = 3.24*10**(-23)
Conversion_kg_Solar_mass = 5.027*10**(-31)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#Hubble Rate
def H(z):
    return H0*(Omega_m*(1+z)**3 + Omega_lambda + Omega_k*((1+z)**2))**(1/2)    
#-----------------------------------------------------------------------------

def Get_me_M_min(BOX_LEN, DIM):
    #-----------------------------------------------------------------------------
    #Setting up initial conditions

    initial_conditions = p21c.initial_conditions(
        user_params = {"DIM": DIM , "HII_DIM": Parameters['HII_DIM'], "BOX_LEN": BOX_LEN},
        cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, OMm = 0.3, OMb = 0.045),
        random_seed=54321
        )
        
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    #Setting up the ionized box at the given redshift and initial conditions

    perturbed_field = p21c.perturb_field(
        redshift = Parameters['z'],
        init_boxes = initial_conditions
        )

    avg_den = np.mean(perturbed_field.density)
    

    rho_c = ((3*H(Parameters['z'])**2)/(8*np.pi*G))*(Conversion_kg_Solar_mass/Conversion_m_to_Mpc)
    rho_avg = Omega_m*rho_c*(avg_den+1)  #Msun/Mpc^3

    M_pixels = rho_avg*(Parameters['BOX_LEN']/(Parameters['DIM']*(1+Parameters['z'])))**3
    
    return M_pixels


if __name__ == '__main__':
    
    M_p = Get_me_M_min(Parameters['BOX_LEN'],Parameters['DIM'])

    print("{:.2e}".format(M_p))
    

