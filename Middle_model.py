#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 05:42:35 2023

@author: sharma

This code calculates all the face models
"""
import numpy as np
import pickle
import sys
from tqdm import tqdm
# To monitor the processing speed
import time
from tabulate import tabulate

#-----------------------------------------------------------------------------
# Damping wing calculation codes
import Calculating_M_pixels as MP
import Parameters_file as Params
from Constants import *
import Generating_ionized_boxes_git as GIB      # This code generates ionized boxes of the given parameters and initial conditions
import Calculating_skewers_git as CS        # This code calculates the neutral fraction weighted over density from different halos along some random sightlines for a given ionized box
import Damping_wings_git as DW         # For a given sightline, it calculates the damping wing profile for a specific halo mass host of a quasar
import Plotting_wings as PW
import Calculating_M_pixels as MP
#-----------------------------------------------------------------------------

def get_middle_model():
    '''
    This funcations calculates the default/middle model with central parameters. This lies at the center of the parametere space

    Returns
    -------
    models_processing_time : float
        time taken to calculate the model

    '''
    
    start_time = time.perf_counter()   # Timer to calculate the calculation time for the mdoels                
       
    print("Calculating the middle model")
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    Parameters = Params.Parameters

    print(f"\nRunning the middle Parametes : ",Parameters)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Calling the Generate Ionized box function
    GIB.Generate_ion_boxes()   # Generating the ionized box
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # Skewers calculations
    print("\nLoading the halo files and the ionized box")
    # Loading all the halos with their masses and coords, and the ionized and desnity of the box
    halo_mass = pickle.load(open(f"{newpath}/Halo_masses_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
    halo_coords = pickle.load(open(f"{newpath}/Halo_coords_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
    ionised_box = pickle.load( open(f"{newpath}/Ionized_box_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))
    density_field = pickle.load( open(f"{newpath}/Density_field_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))
    
    #Picking up all the masss bins
    Mass_bins = np.unique(halo_mass)
    n_Mass_bins = len(Mass_bins)
    print("Mass bins: ", Mass_bins)
    
    file = open(f"{newpath}/Halos_for_skewers_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.txt", 'w')
    
    halo_data_columns = ["Base", "Order", "No. of halos", "Halo Mass"]
    halo_data = []
    
    print("\nCalculating the sightlines")
    for i in tqdm(range(0,n_Mass_bins,int(n_Mass_bins/5))):
        m = (halo_mass == Mass_bins[i])     # Selecting only the halos of the desired mass
        new_halo_mass = halo_mass[m]        # Storing all those halos as new_halo_mass
        new_halo_coords = halo_coords[m]    # Storing their corresponding coordinates in new_halo_coords
        n_halos = len(new_halo_mass)        # Number of halos in the selected mass bin
        o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))             # The order of the mass of the halo in the form of Mass = base*10^order, this is an approximation not the exact value of the masses
        base_halo_mass = int(np.round(new_halo_mass[0]/(10**o_halo_mass),0))        # The base from Mass = base*10^order, base and order are used to seperate halos of different masses 
        
        halo_data.append([base_halo_mass, o_halo_mass, n_halos, np.log10(new_halo_mass[0])])
        
        file.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
        
        # Calling the calculate skewer function
        CS.Calculate_skewers(base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)

    if (n_Mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
        m = (halo_mass == Mass_bins[n_Mass_bins-1])
        new_halo_mass = halo_mass[m]
        new_halo_coords = halo_coords[m]
        n_halos = len(new_halo_mass)
        o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
        base_halo_mass = int(np.round(new_halo_mass[0]/(10**o_halo_mass),0))
        halo_data.append([base_halo_mass, o_halo_mass, n_halos, np.log10(new_halo_mass[0])])
        file.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}s")
    
    print(tabulate(halo_data, headers=halo_data_columns)) 

    # Calling the calculate skewer function
    CS.Calculate_skewers(base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)
    file.close()
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # Damping wings calculations   
    base = []
    order = []
    num_halos = []
    mass_halos = []
    file = open(f"{newpath}/Halos_for_skewers_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.txt",'r')
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
    
    print('Calculating Damping wings')
    # Calling the Damping wings function
    for i in tqdm(range(0,len(base))):
        DW.Damping_Wings(base[i], order[i])  # Calculating damping wings for different halo mass
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # Plotting damping wings
    PW.Plot_wings(base, order, mass_halos, num_halos)
    
    # Time taken to run and plot all the corner models
    models_processing_time = time.perf_counter() - start_time    
    
    return models_processing_time
        
if __name__ == '__main__':

    model_time = get_middle_model()      
    
    