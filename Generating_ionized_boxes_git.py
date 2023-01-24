#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:58:07 2022

@author: sharma

Description: This code generates the ionised boxes calibrated at certain neutral hydrogen fraction. The calibration here is modified by the number of Pop2.ion to make sure the calibration is possible at all redshifts
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

import importlib

#To calculate the code processing time
import time
time_start = time.perf_counter()

#To access pop2 ionisiation photons
from py21cmfast import global_params


# Constants
from Constants import *

#----------------------------------------------------------------------------------------------------------------------------------------------------------

# cache_file = r'/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache' 
# if not os.path.exists(cache_file):
#     os.makedirs(cache_file)
    
# p21c.config['direc'] = cache_file
#cache_tools.clear_cache(direc="_cache")

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#Setting up flags

flags = p21c.inputs.FlagOptions(
    #USE_HALO_FIELD = True,   # To use halo field while updating ionised box
    M_MIN_in_Mass = True,    # To enable the lower mass limit on halos 
    USE_MASS_DEPENDENT_ZETA = True     # To use M_MIN_in_Mass
    )

#-----------------------------------------------------------------------------
    

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def Calibrate(HII_f_esc, perturbed_field, Parameters):
    
    '''
    
    Description
    ----------
    Calibrates the value of HII fraction or f_esc to get the desired initial average neutral Hydrogen fraction of the box
    
    Parameters
    ----------
    HII_f_esc : float
        Takes up the value of HII efficiency/ f_escape factor to calibrate the value of initial neutral H fraction depending upon the value of counter
    
     
    perturbed_field : 
        
    Parameterse : Dictionary 
        Provides the list of parameters to run the ionized box
          
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
    #print(np.mean(ionized_field.xH_box))
    return np.mean(ionized_field.xH_box) - Parameters['target_xh']

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def Calibrate_pop2(pop2, f_esc, perturbed_field, Parameters):
    
    '''
    
    Description
    ----------
    Calibrates the value of Pop2 ionising photons to get the desired initial average neutral Hydrogen fraction of the box
    
    Parameters
    ----------
    pop2 : float
        Takes up the value of Pop2 ionising photons to calibrate the value of initial neutral H fraction depending upon the value of counter
        
    f_esc = float
        Escape fraction of photons escaping from halo
     
    perturbed_field : 
        The ionized box at the required redshift provided by the simulation
        
    Parameterse : Dictionary 
        Provides the list of parameters to run the ionized box
          
    Returns
    -------
    float
        Returns the value of mean neutral H value of the box - target neutral fraction

    '''
    with global_params.use(Z_HEAT_MAX=25):
        p21c.global_params.Pop2_ion = pop2
        
        ionized_field = p21c.ionize_box(
            perturbed_field = perturbed_field,
            astro_params = p21c.AstroParams({"ION_Tvir_MIN":Parameters['T_vir'], "M_TURN":Parameters['M_min'], "F_ESC10":f_esc, "F_STAR10":Parameters['f_star'], "ALPHA_ESC":Parameters['alpha_esc'], "ALPHA_STAR":Parameters['alpha_star']}),
            flag_options= flags, 
        )
        #print(np.mean(ionized_field.xH_box))
        return np.mean(ionized_field.xH_box) - Parameters['target_xh']

#----------------------------------------------------------------------------------------------------------------------------------------------------------
  
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def Generate_ion_boxes(rank = -1):
    
    '''
    
    Description
    ----------
    Generates the ionized box for given parameters and initial conditions
    
    Parameters
    ----------
    rank : Integer, Optional
        Tells the code which parameter file to pick. If no rank is provided then it picks the rank -1, which is the default Parameters file.

    Returns
    -------
    None.
    
    '''
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Importing parameters 
    
    if rank>=0:        
        Para = importlib.import_module(f'Parameters_temp_{rank}')
        Parameters = Para.Parameters
        
    if rank<0:
        import Parameters_file as Para
        Parameters = Para.Parameters

    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    print("\nGenerating ionized box for parameters rank: ", rank)    

    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Halo Mass Function(HMF) from HMFCalc to cross check our HMF from 21cmFast
    mf = MassFunction(z = Parameters['z'], Mmin = Parameters['M_min'], Mmax = 12.0)
    print(Parameters)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    #Setting up initial conditions
    print('\nSetting up initial conditions of the box')
    initial_conditions = p21c.initial_conditions(
        user_params = {"DIM": DIM , "HII_DIM": HII_DIM, "BOX_LEN": L_Box},
        cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, OMm = 0.3, OMb = 0.045),
        random_seed=54321
        )
        
    print('\nPlotting high resolution density plot')    
    plotting.coeval_sliceplot(initial_conditions, "hires_density");
    plt.title("High resolution density")
    plt.savefig(f"{plotpath}/hires_density_no_halofield.png")

    #-----------------------------------------------------------------------------
        
    #-----------------------------------------------------------------------------
    #Setting up the ionized box at the given redshift and initial conditions
    
    print("Perturbing the ionized box to be at the required redshift")
    perturbed_field = p21c.perturb_field(
        redshift = Parameters['z'],
        init_boxes = initial_conditions
        )

    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    #Calibration check---
    print("Performing the calibration check")
    f_esc_extreme = [0,-6]
    original_Pop2_ion = global_params.Pop2_ion
    # Checking if for the given set of parameters we can converge to the target xh by just varying f_esc, if not then we will assume f_esc = 0 and vary Pop2 ion number to get the desired target xh
    if (np.sign(Calibrate(f_esc_extreme[0], perturbed_field, Parameters)) == np.sign(Calibrate(f_esc_extreme[-1], perturbed_field, Parameters))):
        '''
        When we were changing the values of Pop2.ion, it was changing the value for all the succeeding models.
        Hence, not only it was getting slower but also inaccurate.

        For this I can either use:
            with global_params.use(Pop2_ion = 5000):
                global_params.Pop2_ion = optimize.brenth(Calibrate_pop2, 0, global_params.Pop3_ion, xtol= 1e-4, args=(f_esc,perturbed_field,Parameters))
                print(global_params.Pop2_ion)
            print(global_params.Pop2_ion)

        This will only change the values of pop2 ion within that specific portion of the code.
        Or I can restore the value of Pop2 ion after generating the ionized box.
        I prefer the latter, as assigning a variable is faster than calling a new function
        '''
        
        print('\tsame sign')
        f_esc = 0
        print(f"\tCalibrating Pop2 ion to get the target xh, with f_esc : {f_esc}")
        
        # Calibrating the Pop2 ion to get the desired target xh
        print('\tOriginal pop2 ion: ', global_params.Pop2_ion)
        global_params.Pop2_ion = optimize.brenth(Calibrate_pop2, 0, global_params.Pop3_ion, xtol= 1e-4, args=(f_esc,perturbed_field,Parameters))
        print('\tnew pop2 ion: ', global_params.Pop2_ion)
    else:
        #Calibrating the f_esc to get the required xH
        print("\tCalibrating f_esc to get xH = ",Parameters['target_xh'])
        
        f_esc = optimize.brenth(Calibrate, 0, -6, xtol= 1e-4, args=(perturbed_field,Parameters))  # Calibrates the value of f_esc for given target average neutral fraction
        print(f'\tf_esc : {f_esc} for the parameters of rank : {rank}')
        print('\tPop2 ion = ',global_params.Pop2_ion)
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
    print("Plotting the density field")
    plotting.coeval_sliceplot(perturbed_field, "density");
    plt.title("Density Field")
    plt.savefig(f"{plotpath}/Density_z_{Parameters['z']}_no_halofield.png")
    
    # Plotting the velocity field
    print("Plotting the velocity field")
    plotting.coeval_sliceplot(perturbed_field, "velocity");
    plt.title("Velocity Field")
    plt.savefig(f"{plotpath}/Velocity_z_{Parameters['z']}_no_halofield.png")
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    #Calculating the ionised box
    print("Updating ionized boxes")
    
    ionized_field = p21c.ionize_box(
        astro_params = p21c.AstroParams({"ION_Tvir_MIN":Parameters['T_vir'], "M_TURN":Parameters['M_min'], "F_ESC10":f_esc, "F_STAR10":Parameters['f_star'], "ALPHA_ESC":Parameters['alpha_esc'], "ALPHA_STAR":Parameters['alpha_star']}),
        flag_options= flags,
        perturbed_field = perturbed_field,
        pt_halos = Updated_Halo_field   #Inclusion or exclusion of this command didn't make much difference on the final ionised box, enabling the halo field flag did
        )
    
    global_xH = np.mean(ionized_field.xH_box)
    print('\tmean xH: ',global_xH)
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # Saving the data
    print("Writing data to files")
        
    # pickle.dump(ionized_field.xH_box,open( f"{newpath}/Ionized_box_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_halofield.p", "wb" ))
    # pickle.dump(perturbed_field.density,open( f"{newpath}/Density_field_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_halofield.p", "wb" ))
    # pickle.dump(Updated_Halo_field.halo_coords,open( f"{newpath}/Halo_coords_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_halofield.p", "wb" ))
    # pickle.dump(Updated_Halo_field.halo_masses,open( f"{newpath}/Halo_masses_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_halofield.p", "wb" ))
        
    # plotting.coeval_sliceplot(ionized_field, "xH_box");
    # plt.savefig(f"{newpath}/Plots/ionised_box_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_halofield.png" )
    
    pickle.dump(ionized_field.xH_box,open( f"{newpath}/Ionized_box_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
    pickle.dump(perturbed_field.density,open( f"{newpath}/Density_field_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
    pickle.dump(Updated_Halo_field.halo_coords,open( f"{newpath}/Halo_coords_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
    pickle.dump(Updated_Halo_field.halo_masses,open( f"{newpath}/Halo_masses_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
    
    print("Plotting the slice of ionized box")    
    plotting.coeval_sliceplot(ionized_field, "xH_box");
    plt.title("Ionized Box Slice")
    plt.savefig(f"{plotpath}/ionised_box_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png" )
    
    
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
    # plt.savefig(f"{newpath}/Halo_mass_function_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_halofield.png")
    plt.savefig(f"{plotpath}/Halo_mass_function_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png")
    plt.show()
    plt.close()
    
    global_params.Pop2_ion = original_Pop2_ion
    print("Pop2 ion", global_params.Pop2_ion,"\n")
    
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
    rank = 1
    start_time = time.perf_counter()
    print(flags)
    Generate_ion_boxes(145)
    time_elapsed = time.perf_counter() - start_time
    print('Processing time for one model = ', time_elapsed)



