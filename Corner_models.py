#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:54:43 2023

@author: sharma

This code calculates all the corner models
"""
import numpy as np
import pickle
import sys
import importlib
from tqdm import tqdm
# To monitor the processing speed
import time
from tabulate import tabulate
import itertools
import json
import os.path

#-----------------------------------------------------------------------------
# Damping wing calculation codes
import Parameters_file as Params
from Constants import *
import Calculating_M_pixels as MP
import Generating_ionized_boxes_git as GIB      # This code generates ionized boxes of the given parameters and initial conditions
import Calculating_skewers_git as CS        # This code calculates the neutral fraction weighted over density from different halos along some random sightlines for a given ionized box
import Damping_wings_git as DW         # For a given sightline, it calculates the damping wing profile for a specific halo mass host of a quasar
import Plotting_wings as PW
#-----------------------------------------------------------------------------

def get_corner_models(Param_Ranges):
    '''
    This funcations takes the range of parameters as input and calulates the models residing at the corners of the Param_Ranges cube in the parameters space

    Parameters
    ----------
    Param_Ranges : Dictionary
        Provides the range of parameters over which we calculate our models

    Returns
    -------
    models_processing_time : float
        time taken to calculate all the corner models over the given range of parameters

    '''
    
    print(GIB.flags)
    
    print('Writing Parameter files')
    
    variable_keys = [k for k in Param_Ranges]
    variable_keys = [k.replace('_list','') for k in variable_keys] # removing _list from the variable names
    variable_values = [v for v in Param_Ranges.values()]
    # using itertools.product(), to compute all possible permutations
    res = list(itertools.product(*variable_values))
    rank = len(res)  # Total number of combinations
    for i in range(0,rank):
        variables = dict(zip(variable_keys,res[i]))
        p = {p:Params.Parameters[p] for p in Params.Parameters if p not in variables}
        variables = {**variables,**p}
        
        Omega_m_z = (Omega_m*(1+variables['z'])**3)/(Omega_m*(1+variables['z'])**3 + Omega_lambda)                                 
        d = Omega_m_z**2 -1
        Delta_c = 18*np.pi**2 +82*d -39*d**2
        mu = variables['target_xh']*0.5 + (1-variables['target_xh'])
        T_vir = (1.98*10**4)*(mu/0.6)*((10**(variables['M_min'])*h/10**8)**(2/3))*(Omega_m*Delta_c/(Omega_m_z*18*np.pi**2))*((1+variables['z'])/10)
        
        variables['T_vir'] = float('{:.2f}'.format(np.log10(T_vir)))
        with open(f'Parameters_temp_{i}.py', 'w') as f:
            f.write('Parameters = ')
            f.write(json.dumps(variables))
    
    print('\nTotal number of ranks : ',rank) 

    print('\nWorking on Corner Models')                
    start_time = time.perf_counter()    # Timer to calculate the calculation time for the mdoels                
       
    for I in tqdm(range(0,rank)):
        #--------------------------------------------------------------------------------------------------------------------------------------------------------
        Para = importlib.import_module(f'Parameters_temp_{I}')
        Parameters = Para.Parameters

        print(f"\nRunning model no. {I}\n Parametes : ",Parameters)
        #--------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Calling the Generate Ionized box function 
        GIB.Generate_ion_boxes(I)   # Generating the ionized box
        
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
            CS.Calculate_skewers(base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field, I)

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
        CS.Calculate_skewers(base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field, I)
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
            DW.Damping_Wings(base[i], order[i], I)  # Calculating damping wings for different halo mass
        
        #--------------------------------------------------------------------------------------------------------------------------------------------------------
        # Plotting damping wings
        PW.Plot_wings(base, order, mass_halos, num_halos, I)
        
    # Time taken to run and plot all the corner models
    models_processing_time = time.perf_counter() - start_time    
    
    return models_processing_time


#--------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    M_p = MP.Get_me_M_min()
    M_min = float('{:.2f}'.format(np.log10(M_p*20)))
    Param_Ranges ={
        
        'z_list': np.linspace(6,8,2),    # Redshift
        #'M_min_list': np.linspace(M_min,M_min+1,2),    # Minimum mass of star forming halos in base log10
        'target_xh_list': np.linspace(0.1,0.9,2),     # Mean neutral fraction of the box
        'alpha_esc_list': np.linspace(-1,0,2),    # alpha escape
        'alpha_star_list': np.linspace(0,1,2),    # alpha star
        'f_star_list': np.linspace(-2,-0.25,2),   # f star
        'tq_list': np.linspace(0,3.154*10**13,2)     # Quasar lifetime
        
        }

    #print(Param_Ranges['M_min_list'])
    model_time = get_corner_models(Param_Ranges)
    
    
    
'''
# Residue Codes
# This section is a tribute to my attempts for solving different sections in a more cumbersome way

# for I in tqdm(range(0,len(Param_Ranges['z_list']))):
#     for J in range(0,len(Param_Ranges['target_xh_list'])):
#         for K in range(0,len(Param_Ranges['M_min_list'])):
#             for L in range(0,len(Param_Ranges['alpha_esc_list'])):
#                 for M in range(0,len(Param_Ranges['alpha_star_list'])):
#                     for N in range(0,len(Param_Ranges['f_star_list'])):
#                         for O in range(0,len(Param_Ranges['tq_list'])):
#                             Omega_m_z = (Omega_m*(1+Param_Ranges['z_list'][I])**3)/(Omega_m*(1+Param_Ranges['z_list'][I])**3 + Omega_lambda)
                            
#                             d = Omega_m_z**2 -1
#                             Delta_c = 18*np.pi**2 +82*d -39*d**2
#                             mu = Param_Ranges['target_xh_list'][J]*0.5 + (1-Param_Ranges['target_xh_list'][J])
                            
#                             T_vir = (1.98*10**4)*(mu/0.6)*((10**(Param_Ranges['M_min_list'][K])*h/10**8)**(2/3))*(Omega_m*Delta_c/(Omega_m_z*18*np.pi**2))*((1+Param_Ranges['z_list'][I])/10)
#                             variables = {
#                                 'z': Param_Ranges['z_list'][I],
#                                 'target_xh': Param_Ranges['target_xh_list'][J],
#                                 'M_min':Param_Ranges['M_min_list'][K],
#                                 'T_vir':'{:.2f}'.format(np.log10(T_vir)),   # Virial Temperature in base log10
#                                 'alpha_esc': Param_Ranges['alpha_esc_list'][L],
#                                 'alpha_star': Param_Ranges['alpha_star_list'][M],
#                                 'f_star': Param_Ranges['f_star_list'][N],
#                                 'tq': Param_Ranges['tq_list'][O]
#                                 }
                            
#                             rank = 1*O + 2*N + 4*M + 8*L + 16*K + 32*J + 64*I
                            
#                             #print(rank)

#                             f = open(f'/Users/sharma/work/21cmFast_codes_and_plots/Modified_Calibration/Parameters_temp_{rank}.py', 'w')
#                             f.write('Parameters = {\n')

#                             for key, value in variables.items():
#                                 f.write(f"\t'{key}':{variables[key]},\n")
                                        
#                             for p in Params.Parameters:
#                                 if (p not in variables):
#                                     f.write(f"\t'{p}':{Params.Parameters[p]},\n")
#                             f.seek(0,2)
#                             f.seek(f.tell() -2,0)
#                             f.truncate()
#                             f.write('}')
#                             f.close()

'''
