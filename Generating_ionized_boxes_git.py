#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:58:07 2022

@author: sharma

cleaned up version of codes
"""

#-----------------------------------------------------------------------------
#Header Files
import matplotlib.pyplot as plt
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

#HMF class
from hmf import MassFunction 

import pickle

from scipy import optimize

from py21cmfast import global_params

#To calculate the code processing time
import time
time_start = time.perf_counter()

#For new path
import datetime
#----------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------
today = str(datetime.date.today())

# New path for the code in the Astronode

# cache_file = r'/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache' 
# if not os.path.exists(cache_file):
#     os.makedirs(cache_file)
    
# p21c.config['direc'] = cache_file
#cache_tools.clear_cache(direc="_cache")

# Setting up Cache directory for Astronode

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
    

#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing parameters 

from Parameters_temp import Parameters

# Customising some parameters
# Parameters['DIM'] = 512
# Parameters['HII_DIM'] = 128
# Parameters['BOX_LEN'] = 100
# Parameters['target_xh'] = 0.25

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Halo Mass Function from HMFCalc to cross check our HMF from 21cmFast
mf = MassFunction(z = Parameters['z'], Mmin = Parameters['M_min'], Mmax = 12.0)

#----------------------------------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------
#Setting up flags

flags = p21c.inputs.FlagOptions(
    #USE_HALO_FIELD = True,   # To use halo field while updating ionised box
    M_MIN_in_Mass = True,    # To enable the lower mass limit on halos 
    USE_MASS_DEPENDENT_ZETA = True     # To use M_MIN_in_Mass
    )

#-----------------------------------------------------------------------------
    
#-----------------------------------------------------------------------------
#Setting up initial conditions

initial_conditions = p21c.initial_conditions(
    user_params = {"DIM": Parameters['DIM'] , "HII_DIM": Parameters['HII_DIM'], "BOX_LEN": Parameters['BOX_LEN']},
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, OMm = 0.3, OMb = 0.045),
    random_seed=54321
    )
    
    
plotting.coeval_sliceplot(initial_conditions, "hires_density");

#-----------------------------------------------------------------------------
    
#-----------------------------------------------------------------------------
#Setting up the ionized box at the given redshift and initial conditions

perturbed_field = p21c.perturb_field(
    redshift = Parameters['z'],
    init_boxes = initial_conditions
    )

#-----------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def Calibrate(HII_f_esc):
    
    '''
    
    Description
    ----------
    Calibrates the value of HII fraction or f_esc to get the desired initial average neutral Hydrogen fraction of the box
    
    Parameters
    ----------
    HII_f_esc : float
        Takes up the value of HII efficiency/ f_escape factor to calibrate the value of initial neutral H fraction depending upon the value of counter
    
           
    Returns
    -------
    float
        Returns the value of mean neutral H value of the box - target neutral fraction

    '''
    ionized_field = p21c.ionize_box(
        perturbed_field = perturbed_field,
        astro_params = p21c.AstroParams({"ION_Tvir_MIN":Parameters['T_vir'], "M_TURN":Parameters['M_min'], "F_ESC10":HII_f_esc, "F_STAR10":Parameters['f_star'], "ALPHA_ESC":Parameters['alpha_esc'], "ALPHA_STAR":Parameters['alpha_star']}),
        flag_options= flags, 
    )
    return np.mean(ionized_field.xH_box) - Parameters['target_xh']

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def Generate_ion_boxes(newpath):
    
    '''
    
    Description
    -------
    Generates the ionized box for given parameters and initial conditions
    
    Parameters
    ----------
    newpath : string
        Tells the path at which the models will be stored

    Returns
    -------
    None.
    

    '''
    #-----------------------------------------------------------------------------
    #Calibrating the box to get the required xH
    print("Calibrating HII Eff to get xH = ",Parameters['target_xh'])
    
    f_esc = optimize.brenth(Calibrate, 1, -6)  # Calibrates the value of f_esc for given target average neutral fraction

    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # Locating halos
    print("working on Halo List")
    
    Halo_field = p21c.determine_halo_list(
        redshift = Parameters['z'],
        init_boxes = initial_conditions,
        astro_params = p21c.AstroParams({"ION_Tvir_MIN":Parameters['T_vir'], "M_TURN":Parameters['M_min'], "F_ESC10":f_esc, "F_STAR10":Parameters['f_star'], "ALPHA_ESC":Parameters['alpha_esc'], "ALPHA_STAR":Parameters['alpha_star']}),
        flag_options= flags
        )
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # Distributing the halos within the box
    print("Perturbing Halo Field")
    
    Updated_Halo_field = p21c.perturb_halo_list(
        redshift = Parameters['z'],
        init_boxes = initial_conditions,
        halo_field = Halo_field,
        astro_params = p21c.AstroParams({"ION_Tvir_MIN":Parameters['T_vir'], "M_TURN":Parameters['M_min'], "F_ESC10":f_esc, "F_STAR10":Parameters['f_star'], "ALPHA_ESC":Parameters['alpha_esc'], "ALPHA_STAR":Parameters['alpha_star']}),
        flag_options= flags
        )   
    #-----------------------------------------------------------------------------
    
    # Plotting the density field
    plotting.coeval_sliceplot(perturbed_field, "density");
    
    # Plotting the velocity field
    plotting.coeval_sliceplot(perturbed_field, "velocity");
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    #Calculating the ionised box
    print("Updating ionized boxes")
    
    ionized_field = p21c.ionize_box(
        astro_params = p21c.AstroParams({"ION_Tvir_MIN":Parameters['T_vir'], "M_TURN":Parameters['M_min'], "F_ESC10":f_esc, "F_STAR10":Parameters['f_star'], "ALPHA_ESC":Parameters['alpha_esc'], "ALPHA_STAR":Parameters['alpha_star']}),
        flag_options= flags,
        perturbed_field = perturbed_field
        #pt_halos = Updated_Halo_field   #Inclusion or exclusion of this command didn't make much difference on the final ionised box, enabling the halo field flag did
        )
    
    global_xH = np.mean(ionized_field.xH_box)
    print(global_xH,' mean xH')
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # Saving the data
    print("Writing data to files")
        
    pickle.dump(ionized_field.xH_box,open( f"{newpath}/Ionized_box_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
    pickle.dump(perturbed_field.density,open( f"{newpath}/Density_field_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
    pickle.dump(Updated_Halo_field.halo_coords,open( f"{newpath}/Halo_coords_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
    pickle.dump(Updated_Halo_field.halo_masses,open( f"{newpath}/Halo_masses_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
        
    plotting.coeval_sliceplot(ionized_field, "xH_box");
    plt.savefig(f"{newpath}/Plots/ionised_box_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png" )
    
    # Comparing 21cmfast halo mass function with HMFCalc halo mass function
    print("plotting the halo mass function")
    plt.figure(figsize=(5, 6), dpi=150)
    plt.title("Halo Mass function")
    plt.plot(mf.m,mf.dndlnm)
    plt.plot(Halo_field.mass_bins,Halo_field.dndlm)
    plt.xscale('log')
    plt.yscale('log')

    plt.legend(['Hmf_calc','21cmFast'])
    plt.xlabel(r"Mass, $[h^{-1}M_\odot]$")
    plt.ylabel(r"$dn/dm$, $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$");
    #plt.savefig(f"{newpath}/Halo_mass_function_T_vir_{T_vir}.png")
    plt.show()
    plt.close()
    
    #Plotting lightcone
    
    # lightcone = p21c.run_lightcone(
    # redshift = Parameters['z'],
    # max_redshift = 12.0,
    # user_params = {"HII_DIM":Parameters['HII_DIM'], "BOX_LEN": Parameters['BOX_LEN']},
    # lightcone_quantities=("brightness_temp", 'density','xH_box'),
    # global_quantities=("brightness_temp", 'density', 'xH_box'),
    # direc='_cache'
    #            )
    
    # plotting.lightcone_sliceplot(lightcone, "density")
    # #filename = lightcone.save(direc=f'{newpath}')
    # plotting.lightcone_sliceplot(lightcone, "xH_box")
    # #filename2 = lightcone.save(direc=f'{newpath}')
#----------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    Generate_ion_boxes(newpath)



