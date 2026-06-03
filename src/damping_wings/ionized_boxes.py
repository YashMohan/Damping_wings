#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:58:07 2022

@author: sharma

Description: This code generates the ionised boxes calibrated at certain neutral hydrogen fraction. The calibration here is modified by the number of Pop2.ion to make sure the calibration is possible at all redshifts
"""

#-----------------------------------------------------------------------------
import matplotlib.pyplot as plt
from typing import TypedDict
import sys
import py21cmfast as p21c
from py21cmfast import plotting
import numpy as np
from hmf import MassFunction 
import pickle
from scipy import optimize
import time
time_start = time.perf_counter()
p21c.config['EXTRA_HALOBOX_FIELDS'] = True

from .config.constants import N_sightlines, newpath, L_Box, HII_DIM, DIM, txt_files, seed
from .config.parameters_file import Parameters as params 
#-----------------------------------------------------------------------------
class SimParams(TypedDict):
    x_hi: float
    m_min: float
    t_q: float
    m_qso: float
    redshift: float

#-----------------------------------------------------------------------------
    

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def calibrate(HII_f_esc: float, Parameters: SimParams, initial_conditions: p21c.InitialConditions) -> float:
    
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

    initial_conditions : p21c.InitialConditions
        Initial conditions from the simulation box
          
    Returns
    -------
    float
        Returns the difference vetween the mean neutral hydrogen fraction from the box and the target neutral fraction

    '''

    new_inputs: p21c.InputParameters    # Updating input parameters for calibration
    new_initial_conditions: p21c.InitialConditions  # Updating initial conditions for calibration
    perturbed_field: p21c.PerturbedField # Updating the density fields perturbed to the desired redshift
    ionized_field: p21c.IonizedBox # Calculating the ionized fields for calibration

    new_inputs = initial_conditions.inputs.evolve_input_structs(ION_Tvir_MIN = Parameters['T_vir'], M_TURN = Parameters['m_min'], F_ESC10 = HII_f_esc, F_STAR10 = Parameters['f_star'], ALPHA_ESC = Parameters['alpha_esc'], ALPHA_STAR = Parameters['alpha_star'])
    new_initial_conditions = p21c.compute_initial_conditions(inputs=new_inputs)

    perturbed_field = p21c.perturb_field(
        redshift = Parameters['z'],
        initial_conditions = new_initial_conditions
        )

    ionized_field = p21c.compute_ionization_field(
    initial_conditions=new_initial_conditions, perturbed_field=perturbed_field
    )
    
    return ionized_field.get("neutral_fraction").mean() - Parameters['target_xh']

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def calibrate_pop2(pop2: float, f_esc: float, Parameters: SimParams, initial_conditions: p21c.InitialConditions) -> float:
    
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
        
    Parameterse : Dictionary 
        Provides the list of parameters to run the ionized box

    initial_conditions : p21c.InitialConditions
        Initial conditions from the simulation box
          
    Returns
    -------
    float
        Returns the difference vetween the mean neutral hydrogen fraction from the box and the target neutral fraction

    '''
    new_inputs: p21c.InputParameters    # Updating input parameters for calibration
    new_initial_conditions: p21c.InitialConditions  # Updating initial conditions for calibration
    perturbed_field: p21c.PerturbedField # Updating the density fields perturbed to the desired redshift
    ionized_field: p21c.IonizedBox # Calculating the ionized fields for calibration

    new_inputs = initial_conditions.inputs.evolve_input_structs(POP2_ION = pop2, new_inputs = initial_conditions.inputs.evolve_input_structs(ION_Tvir_MIN = Parameters['T_vir'], M_TURN = Parameters['m_min'], F_ESC10 = f_esc, F_STAR10 = Parameters['f_star'], ALPHA_ESC = Parameters['alpha_esc'], ALPHA_STAR = Parameters['alpha_star']))
    new_initial_conditions = p21c.compute_initial_conditions(inputs=new_inputs)

    perturbed_field = p21c.perturb_field(
        redshift = Parameters['z'],
        initial_conditions = new_initial_conditions
        )
    
    ionized_field = p21c.compute_ionization_field(
        perturbed_field = perturbed_field,
        initial_conditions = new_initial_conditions
        )

    return ionized_field.get("neutral_fraction").mean() - Parameters['target_xh']

#----------------------------------------------------------------------------------------------------------------------------------------------------------
  
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_ion_boxes(initial_conditions: p21c.InitialConditions, cache:  p21c.OutputCache, Parameters: SimParams, rank: int) -> None:
    
    '''
    
    Description
    ----------
    Generates the ionized box for given parameters and initial conditions
    
    Parameters
    ----------
    initial_conditions : p21c.InitialConditions
        Initial conditions from the simulation box

    cache :  p21c.OutputCache
        Cache object for 21cmFAST

    Parameterse : Dictionary 
        Provides the list of parameters to run the ionized box
    
    rank : Integer, Optional
        Tells the code which parameter file to pick.

    Returns
    -------
    None.
    
    '''

    mf : MassFunction   # Halo Mass Function(HMF) from HMFCalc to cross check our HMF from 21cmFast
    f_esc_extreme: list[int]   # Extremes of escape fraction
    original_Pop2_ion: float    # 5000, default POP2 ION value from 21cmFAST
    f_esc: float    # Escape fraction
    calibrate_pop2_ion: float   # Calibrated POP2 ION value if calibrate_pop2 function is used
    new_inputs: p21c.InputParameters    # Updating input parameters for calibration
    new_initial_conditions: p21c.InitialConditions  # Updating initial conditions for calibration
    perturbed_field: p21c.PerturbedField    # Updating the density fields perturbed to the desired redshift
    ionized_field: p21c.IonizedBox  # Calculating the ionized fields for calibration
    Halo_field: p21c.HaloCatalog    # Halo Catalog 
    Updated_Halo_field: p21c.HaloBox    # Calculating the perturbed Halo properties


    print("\nGenerating ionized box for parameters rank: ", rank)    

    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    mf = MassFunction(z = Parameters['z'], Mmin = Parameters['m_min'], Mmax = 12.0,hmf_model="SMT")
    print(Parameters)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    #Setting up the ionized box at the given redshift and initial conditions    
    #-----------------------------------------------------------------------------
    
    #Calibration check---
    print("Performing the calibration check")
    f_esc_extreme = [0,-6]
    original_Pop2_ion = initial_conditions.astro_params.POP2_ION
    # Checking if for the given set of parameters we can converge to the target xh by just varying f_esc, if not then we will assume f_esc = 0 and vary Pop2 ion number to get the desired target xh
    if (np.sign(calibrate(f_esc_extreme[0], Parameters, initial_conditions)) == np.sign(calibrate(f_esc_extreme[-1], Parameters, initial_conditions))):
        
        print('\tsame sign')
        f_esc = 0
        new_inputs = initial_conditions.inputs.evolve_input_structs(F_ESC10 = f_esc)
        new_initial_conditions = initial_conditions.new(inputs=new_inputs)
        print(f"\tCalibrating Pop2 ion to get the target xh, with f_esc : {f_esc}")
        
        # Calibrating the Pop2 ion to get the desired target xh
        print('\tOriginal pop2 ion: ', original_Pop2_ion)
        calibrate_pop2_ion = optimize.brenth(calibrate_pop2, 0, new_initial_conditions.astro_params.POP3_ION, xtol= 1e-4, args=(f_esc,Parameters, initial_conditions))
        print('\tnew pop2 ion: ', calibrate_pop2_ion)
    
    else:
        #Calibrating the f_esc to get the required xH
        print("\tCalibrating f_esc to get xH = ",Parameters['target_xh'])
        calibrate_pop2_ion = original_Pop2_ion
        f_esc = optimize.brenth(calibrate, 0, -6, xtol= 1e-4, args=(Parameters, initial_conditions))  # Calibrates the value of f_esc for given target average neutral fraction
    
    print(f'\tf_esc : {f_esc} for the parameters of rank : {rank}')
    print('\tPop2 ion = ',calibrate_pop2_ion)
    
    with open(f'{txt_files}/Additional_data_{rank}_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.txt', 'w') as f:
        f.write(f'f_esc {f_esc} \n')
        f.write(f'Pop2 ion {calibrate_pop2_ion}')
    #-----------------------------------------------------------------------------
 
    #-----------------------------------------------------------------------------
    # Locating halos

    new_inputs = initial_conditions.inputs.evolve_input_structs(POP2_ION = calibrate_pop2_ion, ION_Tvir_MIN = Parameters['T_vir'], M_TURN = Parameters['m_min'], F_ESC10 = f_esc, F_STAR10 = Parameters['f_star'], ALPHA_ESC = Parameters['alpha_esc'], ALPHA_STAR = Parameters['alpha_star'])
    new_initial_conditions = p21c.compute_initial_conditions(inputs=new_inputs)

    perturbed_field = p21c.perturb_field(
        redshift = Parameters['z'],
        initial_conditions = new_initial_conditions
        )
    
    ionized_field = p21c.compute_ionization_field(
        perturbed_field = perturbed_field,
        initial_conditions = new_initial_conditions,
        inputs = new_inputs,
        cache = cache,
        write = True
        )
    
    print("working on Halo List")
    
    Halo_field = p21c.determine_halo_catalog(
        redshift = Parameters['z'],
        initial_conditions = new_initial_conditions,
        cache = cache,
        write = True
        )
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # Distributing the halos within the box
    print("Perturbing Halo Field")
    
    Updated_Halo_field = p21c.compute_halo_grid(
        inputs = new_inputs,
        redshift = Parameters['z'],
        initial_conditions = new_initial_conditions,
        halo_catalog = Halo_field,
        previous_ionize_box = ionized_field,
        cache = cache,
        write = True
        )   

    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    """
    # Plotting the density field
    print("Plotting the density field")
    plotting.coeval_sliceplot(perturbed_field, "density");
    plt.title("Density Field")
    plt.savefig(f"{plotpath}/Density_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.png")
    
    # Plotting the velocity field
    print("Plotting the velocity field")
    plotting.coeval_sliceplot(perturbed_field, "velocity");
    plt.title("Velocity Field")
    plt.savefig(f"{plotpath}/Velocity_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.png")

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
    # plt.savefig(f"{newpath}/Halo_mass_function_rank_{rank}_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.png")
    plt.savefig(f"{plotpath}/Halo_mass_function_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.png")
    plt.show()
    plt.close()
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    """
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    print("Writing data to files")  
    
    pickle.dump(ionized_field.get("neutral_fraction"),open( f"{newpath}/Ionized_box_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "wb" ))
    pickle.dump(perturbed_field.get("density"),open( f"{newpath}/Density_field_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "wb" ))
    pickle.dump(Updated_Halo_field.get("halo_coords"),open( f"{newpath}/Halo_coords_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "wb" ))
    pickle.dump(Updated_Halo_field.get("halo_masses"),open( f"{newpath}/Halo_masses_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "wb" ))
    
    print("Plotting the slice of ionized box")    
    plotting.coeval_sliceplot(ionized_field, "xH_box");
    plt.title("Ionized Box Slice")
    plt.savefig(f"{plotpath}/ionised_box_rank_{rank}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.png" )
#----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    pass



