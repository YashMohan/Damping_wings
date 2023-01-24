#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 05:21:30 2023

@author: sharma
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import pickle
import os
import sys
import importlib
from Constants import *

def Plot_wings(base,order,mass_halos,num_halos,rank = -1):
    '''
    This code plots the damping wing signals from the given list of masses along with their scatter and the difference in scatter.

    Parameters
    ----------
    base : Array
        Base of the values of halo mass, it is provided to seperate halos and thier data files. Mass ~ base*10^order
    order : Array
        Order of the values of halo mass, it is provided to seperate halos and thier data files
    mass_halos : Array
        The masses of the corresponding halos
    num_halos : Array
        The number of halos in the given mass bins
    rank : Integer, Optional
        Tells the code which parameter file to pick. If no rank is provided then it picks the rank -1, which is the default Parameters file.

    Returns
    -------
    None.

    '''
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # Importing parameters 
    
    if rank>=0:
        
        Para = importlib.import_module(f'Parameters_temp_{rank}')
        Parameters = Para.Parameters
        
    if rank<0:
        
        import Parameters_file as Para
        Parameters = Para.Parameters

    print(f"\nPlotting for model no. {rank}\n Parametes : ",Parameters)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #Plottting the damping wings

    lamda = pickle.load(open(f"{newpath}/lamda_z_{Parameters['z']}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

    e_tau_avg = np.zeros((len(base),n_pixels))
    low_quantile = np.zeros((len(base),n_pixels))
    mid_quantile = np.zeros((len(base),n_pixels))
    up_quantile = np.zeros((len(base),n_pixels))
    diff_quantile = np.zeros((len(base),n_pixels))

    plt.figure(figsize=(5, 6), dpi=150)
    ax = plt.axes()
    plt.minorticks_on() 
    ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
    ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
    plt.title(f"Averaged damping wings for various halo masses")

    for i in range(0,len(base)):
        e_tau_avg[i] = (pickle.load(open(f"{newpath}/e_tau_avg_Mass_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        low_quantile[i] = (pickle.load(open(f"{newpath}/lower_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        mid_quantile[i] = (pickle.load(open(f"{newpath}/middle_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        up_quantile[i] = (pickle.load(open(f"{newpath}/upper_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        #tau_z[i] = (pickle.load(open(f"{newpath}/tau_z_all_Mass_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        plt.plot(lamda,e_tau_avg[i], label = f'({mass_halos[i]}, {num_halos[i]})')
        # plt.plot(lamda,low_quantile[i], 'k--')
        # plt.plot(lamda,up_quantile[i], 'k--')
        plt.fill_between(lamda, up_quantile[i], low_quantile[i], alpha=0.2)

    ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
    ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
    plt.xlim(1170,1240)
    plt.tight_layout()
    legend = plt.legend(loc='lower right', fontsize='small')
    plt.savefig(f"{plotpath}/Damping_wings_averaged_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png",  bbox_inches='tight', dpi=1000)
    plt.show()
    plt.close()

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #Plottting the damping wings along with quantiles for low and high mass halos    
    
    for k in range(len(base)-1,0,-1):
        if (float(num_halos[k])>=10.0):
            #print(k)
            break
    plt.figure(figsize=(5, 6), dpi=150)
    ax = plt.axes()
    plt.minorticks_on() 
    ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
    ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)

    plt.plot(lamda,mid_quantile[0], label = f'({mass_halos[0]}, {num_halos[0]})',)
    plt.fill_between(lamda, up_quantile[0], low_quantile[0], alpha=0.25)

    plt.plot(lamda,mid_quantile[k], label = f'({mass_halos[k]}, {num_halos[k]})',)
    plt.fill_between(lamda, up_quantile[k], low_quantile[k], alpha=0.25)

    plt.axvline(x = 1215.67, ymin=0.0, ymax=1.0, color = 'black', linestyle='dashed', label = r"$Ly_{\alpha}$")

    ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
    ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
    plt.xlim(1170,1270)
    plt.tight_layout()
    legend = plt.legend(loc='center right', fontsize='small')
    plt.savefig(f"{plotpath}/quantile_plot_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png", bbox_inches='tight', dpi=1000)
    plt.show()
    plt.close()

    #--------------------------------------------------------------------------------------------------------------------------------------------------------    
    # Plotting the difference between upper and lower quantiles for all halo massses
    
    plt.figure(figsize=(5, 6), dpi=150)
    ax = plt.axes()
    plt.minorticks_on() 
    ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
    ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
    plt.title("Quantile difference")
    
    for i in range(0,len(base)):
        diff_quantile[i] = (pickle.load(open(f"{newpath}/diff_quantile_{base[i]}_{order[i]}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
        plt.plot(lamda,diff_quantile[i], label = f'({mass_halos[i]}, {num_halos[i]})')
    ax.set_ylabel(r'$1-\sigma$',  fontsize=20)
    ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
    plt.xlim(1170,1240)
    plt.yscale("log")
    plt.ylim(10**(-2),1)
    plt.tight_layout()
    legend = plt.legend(loc='lower left', fontsize='small')
    plt.savefig(f"{plotpath}/1-sigma_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png", bbox_inches='tight', dpi=1000)
    plt.show()
    plt.close()
    
    
    #-----------------------------------------------------------------------------
