#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:54:03 2022

@author: sharma
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import datetime
import importlib
from Constants import *
from tqdm import tqdm
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
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

def Damping_Wings(base,order,rank = -1):
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
    # Importing parameters 
    
    if rank>=0:
        
        Para = importlib.import_module(f'Parameters_temp_{rank}')
        Parameters = Para.Parameters
        
    if rank<0:
        
        import Parameters_file as Para
        Parameters = Para.Parameters

    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    tau_gp = (7.16*10**5)*((1+Parameters['z'])/10)**(3/2)

    #Calculating damping wings weighted over density
    
    len_z = n_pixels  # The length of redshift space is same as n_pixels, it allows me to equate tau_z[j] with z while calculating damping wing, hence making the calculation much faster as I'm using the vectors now
    z = np.linspace(Parameters['z']-1.0,Parameters['z']+1.0,num=len_z) #range of z for which the damping wings will be calculated
    delta_z = 1/len_z
    
    tau_avg = np.zeros(n_pixels)       # Averaged damping wing optical depth over all the sightlines
    tau_z = np.zeros((N_sightlines,n_pixels))     # Damping wing optical depth for each sightline
    
    low_quantile = np.zeros(n_pixels)
    mid_quantile = np.zeros(n_pixels)
    up_quantile = np.zeros(n_pixels)
    diff_quantile = np.zeros(n_pixels)
    
    plt.figure(figsize=(5, 6), dpi=150)
    ax = plt.axes()
    plt.minorticks_on() 
    ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
    ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
    plt.title(f"Damping wings for ${base}*10^{order} M\odot$ halos weighted over density")
    
    print('Calculating the optical depth')
    for j in range(0,N_sightlines):  # Looping over all sightlines

        zb = Parameters['z']
        ze = Parameters['z'] - dl*H(Parameters['z'])/(c)  # redshift end points of small patches
        xh = pickle.load( open( f"{newpath}/xh_HM_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_{j}_no_halofield.p", "rb" ) )
        den = pickle.load( open( f"{newpath}/density_HM_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_{j}_no_halofield.p", "rb" ) )
        
        for i in range(1,n_pixels):
            zb = Parameters['z'] - i*dl*H(zb)/(c)  # Upadting zb and ze from patch to patch
            ze = zb - dl*H(zb)/(c)
            # Calculating the optical depth 
            tau_z[j] = tau_z[j] + np.abs((xh[i]*(den[i]+1))*(tau_gp*R_alpha/np.pi)*(((1+zb)/(1+z))**(3/2))*np.abs((np.abs(I((1+zb)/(1+z)))-np.abs(I((1+ze)/(1+z)))))) # Mesinger
        
        tau_z[j][0] = 10    # Settting up the lower end, as for blueward side, as there will only be neutral gas and hence the wings will be damped quickly or e^-tau ~ 0
        for k in range(n_pixels-1,0,-1):    # Cut off for blueward side, once tau_z hits a threshold value the damping drops to ~0 blueward to that
            if(tau_z[j][k]>=5) | (tau_z[j][k] < 0):
                break

        for l in range(k,0,-1):
            tau_z[j][l] = 10 
    
        tau_avg = tau_avg + tau_z[j]/N_sightlines  # Calculating the average damping wing optical depth over all sightlines
        
        e_tau_z = np.exp(-tau_z[j])
        lamda = (1+z)*1215.67/(1+Parameters['z'])   # Observed wavelength
        plt.plot(lamda,e_tau_z)
        
    e_tau_avg = np.exp(-tau_avg)
    
    pickle.dump(lamda,open( f"{newpath}/lamda_z_{Parameters['z']}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "wb" ))
    pickle.dump(e_tau_avg,open(f"{newpath}/e_tau_avg_Mass_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","wb"))
    pickle.dump(tau_z,open(f"{newpath}/tau_z_all_Mass_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","wb"))
    
    plt.plot(lamda,e_tau_avg,'k--' , label = 'Average value')
    ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
    ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
    plt.xlim(1170,1270)
    plt.tight_layout()
    #legend = ax.legend(loc='center right', shadow=True, fontsize='large')
    #legend.get_frame().set_facecolor('C0')
    plt.savefig(f"{newpath}/Plots/Damping_wing_Mass_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png")
    plt.show()
    plt.close()
    
    print('Calculating the quantiles')
    for i in range(0,n_pixels):
        low_quantile[i] = np.quantile(np.exp(-tau_z[:,i]),0.16)
        mid_quantile[i] = np.quantile(np.exp(-tau_z[:,i]),0.50)
        up_quantile[i] = np.quantile(np.exp(-tau_z[:,i]),0.84)
        diff_quantile[i] = up_quantile[i] - low_quantile[i]
        
    pickle.dump(low_quantile,open(f"{newpath}/lower_quantile_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","wb"))
    pickle.dump(mid_quantile,open(f"{newpath}/middle_quantile_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","wb"))
    pickle.dump(up_quantile,open(f"{newpath}/upper_quantile_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","wb"))
    pickle.dump(diff_quantile,open(f"{newpath}/diff_quantile_{base}_{order}_tq_{Parameters['tq']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","wb"))

if __name__ == '__main__':
    

    base = []
    order = []
    num_halos = []
    mass_halos = []
    file = open(f"{newpath}/Halos_for_skewers.txt",'r')
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
    #sys.exit()
    
    for i in range(0,len(base)):
        Damping_Wings(base[i], order[i])



