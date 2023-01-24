#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 05:42:35 2023

@author: sharma

This code calculates all the face models
"""
import numpy as np
import pickle
import os
import sys
import importlib
from tqdm import tqdm
# To monitor the processing speed
import time
from tabulate import tabulate
import json

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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def Comprehensive_plots(d,D):
    '''
    This code generates the comprehensive plot, which compares different models by comparing their damping wing signals and 1-sigma scatter, for face models

    Parameters
    ----------
    d : Integer
        Length of variables list for face models
    D : Integer
        Length of the total parameters list

    Returns
    -------
    None.

    '''
    #---------------------------------------------------------------------------------------------------------------------
    # Comprehensive Plots for face models   
    
    #plt.figure(dpi=5000)
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1],sharex = ax0)

    plt.minorticks_on() 
    #ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
    #ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
    #plt.title(f"Averaged damping wings for various halo masses (xh = {Parameters['target_xh']})")

    for I in range(0,2*d):
        #-----------------------------------------------------------------------------   
        rank = 2**D + 2*I + 1  # Selecting only tq = 10^6 yrs models, +1 for tq = 10^6 yrs, 0 for tq = 0 models
        
        print(rank)
        Para = importlib.import_module(f'Parameters_temp_{rank}')
        Parameters = Para.Parameters
        #-----------------------------------------------------------------------------

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

        #Plottting the damping wings

        lamda = pickle.load(open(f"{newpath}/lamda_z_{Parameters['z']}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

        e_tau_avg = np.zeros((len(base),n_pixels))
        low_quantile = np.zeros((len(base),n_pixels))
        mid_quantile = np.zeros((len(base),n_pixels))
        up_quantile = np.zeros((len(base),n_pixels))
        diff_quantile = np.zeros((len(base),n_pixels))


        for i in range(0,len(base)):
            e_tau_avg[i] = (pickle.load(open(f"{newpath}/e_tau_avg_Mass_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
            low_quantile[i] = (pickle.load(open(f"{newpath}/lower_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
            mid_quantile[i] = (pickle.load(open(f"{newpath}/middle_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
            up_quantile[i] = (pickle.load(open(f"{newpath}/upper_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
            diff_quantile[i] = (pickle.load(open(f"{newpath}/diff_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
         
        for k in range(len(base)-1,0,-1):
            if (float(num_halos[k])>=10.0):
                break

        ax0.plot(lamda,mid_quantile[0], label = f"(xh: {Parameters['target_xh']}, alpha-esc: {Parameters['alpha_esc']}, alpha-star: {Parameters['alpha_star']}, f_star: {Parameters['f_star']}, mass = {mass_halos[0]})")
        ax1.plot(lamda,diff_quantile[0], label = f"(xh: {Parameters['target_xh']}, alpha-esc: {Parameters['alpha_esc']}, alpha-star: {Parameters['alpha_star']}, f_star: {Parameters['f_star']}, mass = {mass_halos[0]})")
        ax0.fill_between(lamda, up_quantile[0], low_quantile[0], alpha=0.25)

        # plt.plot(lamda,mid_quantile[k], label = f"(xh: {Parameters['target_xh']}, alpha-esc: {Parameters['alpha_esc']}, alpha-star: {Parameters['alpha_star']}, f_star: {Parameters['f_star']}, mass = {mass_halos[k]})")
        # plt.fill_between(lamda, up_quantile[k], low_quantile[k], alpha=0.25)

    # Plotting the same for the middle model
    Parameters = Params.Parameters
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------

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

    #Plottting the damping wings

    lamda = pickle.load(open(f"{newpath}/lamda_z_{Parameters['z']}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

    e_tau_avg = np.zeros((len(base),n_pixels))
    low_quantile = np.zeros((len(base),n_pixels))
    mid_quantile = np.zeros((len(base),n_pixels))
    up_quantile = np.zeros((len(base),n_pixels))
    diff_quantile = np.zeros((len(base),n_pixels))


    for i in range(0,len(base)):
        e_tau_avg[i] = (pickle.load(open(f"{newpath}/e_tau_avg_Mass_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        low_quantile[i] = (pickle.load(open(f"{newpath}/lower_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        mid_quantile[i] = (pickle.load(open(f"{newpath}/middle_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        up_quantile[i] = (pickle.load(open(f"{newpath}/upper_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        diff_quantile[i] = (pickle.load(open(f"{newpath}/diff_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
     
    for k in range(len(base)-1,0,-1):
        if (float(num_halos[k])>=10.0):
            #print(k)
            break

    ax0.plot(lamda,mid_quantile[0], color = 'black', linestyle='dashed', linewidth=1.5, label = f"(xh: {Parameters['target_xh']}, alpha-esc: {Parameters['alpha_esc']}, alpha-star: {Parameters['alpha_star']}, f_star: {Parameters['f_star']}, mass = {mass_halos[0]})")
    ax0.fill_between(lamda, up_quantile[0], low_quantile[0], alpha=0.25)
    ax1.plot(lamda,diff_quantile[0], color = 'black', linestyle='dashed', linewidth=1.5, label = f"(xh: {Parameters['target_xh']}, alpha-esc: {Parameters['alpha_esc']}, alpha-star: {Parameters['alpha_star']}, f_star: {Parameters['f_star']}, mass = {mass_halos[0]})")
    # plt.plot(lamda,mid_quantile[k], color = 'black', linestyle='dashed', linewidth=1.5, label = f"(xh: {Parameters['target_xh']}, alpha-esc: {Parameters['alpha_esc']}, alpha-star: {Parameters['alpha_star']}, f_star: {Parameters['f_star']}, mass = {mass_halos[k]})")
    # plt.fill_between(lamda, up_quantile[k], low_quantile[k], alpha=0.25)

    ax0.axvline(x = 1215.67, ymin=0.0, ymax=1.0, color = 'black', linestyle='dashed', label = r"$Ly_{\alpha}$")
    ax0.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=10)
    ax1.set_ylabel(r'$1-\sigma$',  fontsize=10)
    ax1.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=10)
    ax0.set_xlim(1180,1260)
    ax1.set_xlim(1180,1260)
    plt.setp(ax0.get_xticklabels(), visible=False)
    #plt.tight_layout()
    
    pos = ax0.get_position()
    pos2 = ax1.get_position()
    ax0.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
    ax1.set_position([pos2.x0, pos2.y0, pos2.width * 0.8, pos2.height])
    ax0.legend(loc='center right', bbox_to_anchor=(2.5, -0.13))
    #legend = plt.legend(loc='center right', fontsize='small')
    plt.subplots_adjust(hspace=.0)
    plt.savefig(f"{plotpath}/Comprehensive_quantile_plots_tq_{Parameters['tq']}_z_{Parameters['z']}_low_mass_halos_{mass_halos[0]}_calibrated_no_halofield.png",  bbox_inches='tight', dpi=1000)
    plt.show()
    plt.close()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_face_models(variables_list,variables_ranges,D):
    '''
    This funcations takes the range of parameters as input and calulates the models residing at the faces of the variables_Ranges cube in the parameters space

    Parameters
    ----------
    variables_list : List
        List of variables we need to vary for face models
    variables_ranges : List
        The values over which the parameters need to vary
    D : Integer
        The total number of variable parameters. Note, total number of parameters can be different from the list of parameters we are varying in thi program

    Returns
    -------
    models_processing_time : float
        time taken to calculate all the face models over the given range of parameters

    '''
    
    d = len(variables_list)
    
    for I in range(0,2*d):
    
        variables = dict(Params.Parameters)
        q = int(I/2)
        r = int(I%2)
        keys = [k for k in variables if k == variables_list[q]]
        variables[keys[0]] = variables_ranges[q][r]  # Keys[0] is used since keys is a list with only 1 element, which we are varying
        
        Omega_m_z = (Omega_m*(1+variables['z'])**3)/(Omega_m*(1+variables['z'])**3 + Omega_lambda)
        d = Omega_m_z**2 -1
        Delta_c = 18*np.pi**2 +82*d -39*d**2
        mu = variables['target_xh']*0.5 + (1-variables['target_xh'])
        T_vir = (1.98*10**4)*(mu/0.6)*(((10**variables['M_min'])*h/10**8)**(2/3))*(Omega_m*Delta_c/(Omega_m_z*18*np.pi**2))*((1+variables['z'])/10)
 
        variables['T_vir'] = float('{:.2f}'.format(np.log10(T_vir)))
        
        for J in range(1,-1,-1):  # It is a seperate loop since we want to vary tq for all the models
            variables['tq'] = J*variables['tq']  # Works well since tq is either 0 or 10^6 years, for any other values of tq, we can just loop over that list instead of going from 1 to 0 
        
            rank = 2**D + 2*I + J    # odd rank is for tq = 10^6 yrs, even rank is for tq = 0
            with open(f'Parameters_temp_{rank}.py', 'w') as f:
                f.write('Parameters = ')
                f.write(json.dumps(variables))
    rank = rank +2
    
    start_time = time.perf_counter()   # Timer to calculate the calculation time for the mdoels                
       
    print("Calculating Face models")
    for I in tqdm(range(2**D,rank)):
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
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        
if __name__ == '__main__':

    M_p = MP.Get_me_M_min()        # Average pixel mass

    M_min = float('{:.2f}'.format(np.log10(M_p*20)))

    Params.Parameters['M_min'] = M_min

    Param_Ranges ={
        
        'z_list': np.linspace(6,8,2),    # Redshift
        'M_min_list': np.linspace(M_min,M_min+1,2),    # Minimum mass of star forming halos
        'target_xh_list': np.linspace(0.1,0.9,2),     # Mean neutral fraction of the box
        'alpha_esc_list': np.linspace(-1,0,2),    # alpha escape
        'alpha_star_list': np.linspace(0,1,2),    # alpha star
        'f_star_list': np.linspace(-2,-0.25,2),   # f star
        'tq_list': np.linspace(0,3.154*10**13,2)     # Quasar lifetime
        
        }

    D = len(Param_Ranges)  # Dimension of corner models, i.e., total corner models = 2^D, corners of D-dim hypercube
    variables_list = ['M_min', 'target_xh', 'alpha_esc', 'alpha_star', 'f_star']  # List of variables we need for face models
    variables_list2 = [f"{k}_list" for k in variables_list]         # adding _list suffix to compare with Param_Ranges
    variables_ranges = [Param_Ranges[k] for k in variables_list2 if k in Param_Ranges]    # Selecting the parameter ranges of the desired list of variables

    model_time = get_face_models(variables_list, variables_ranges, D)      
    
    