#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:54:03 2022

@author: sharma
"""

import numpy as np
from scipy.integrate import odeint
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import datetime
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
today = str(datetime.date.today())
#newpath = r'/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/'+today 
# if not os.path.exists(newpath):
#     print("Could not find the directory to load data")
#     sys.exit()
# #-----------------------------------------------------------------------------
    

#-----------------------------------------------------------------------------
# New path for the code in the laptop
newpath = r'/Users/sharma/work/21cmFast_codes_and_plots/'+today 

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#Parameters to vary
from Parameters_temp import Parameters

# Customising some parameters
Parameters['DIM'] = 512
Parameters['HII_DIM'] = 128
Parameters['BOX_LEN'] = 100
Parameters['target_xh'] = 0.25

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Constants

H0 = 70000.0    #m/s/Mpc
Omega_m = 0.3
Omega_lambda = 0.7
Omega_k = 0.0
tau_gp = (7.16*10**5)*((1+Parameters['z'])/10)**(3/2)
Lambda = 6.25*10**8 #s^-1
nu_alpha = 2.47*10**15  # hz
R_alpha = Lambda/(4*np.pi*nu_alpha)
c = 3*10**8 #ms^-1
L_Box = 200 #Mpc Co-moving
DIM = 1024
HII_DIM = 256
dl = L_Box/HII_DIM
n_pixels = 300

N = 100 # Number of sightlines
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def I(x):

    return ((x**(1/2))/(1-x) + 2*np.log(np.abs((1-x**(1/2))/(1+x**(1/2)))))
 
def H(z):
    return H0*(Omega_m*(1+z)**3 + Omega_lambda + Omega_k*((1+z)**2))**(1/2)    
#-----------------------------------------------------------------------------

def Damping_Wings(base,order):
    #-----------------------------------------------------------------------------
    
    #Calculating damping wings weighted over density
    
    len_z = n_pixels
    z = np.linspace(Parameters['z']-1.0,Parameters['z']+1.0,num=len_z) #range of z for which the damping wings will be calculated
    delta_z = 1/(n_pixels-1)
    
    tau_avg = np.zeros(n_pixels)
    tau_z = np.zeros((N,n_pixels))
    tau_scatter = np.zeros((N,n_pixels))
    
    plt.figure(figsize=(5, 6), dpi=150)
    ax = plt.axes()
    plt.minorticks_on() 
    ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
    ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
    plt.title(f"Damping wings for ${base}*10^{order} M\odot$ halos weighted over density")
    
    
    for j in range(0,N):
        
        zb = Parameters['z']
        ze = Parameters['z'] - dl*H(Parameters['z'])/(c)
        xh = pickle.load( open( f"{newpath}/xh_HM_{base}_{order}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_z_{Parameters['z']}_calibrated_{j}.p", "rb" ) )
        den = pickle.load( open( f"{newpath}/density_HM_{base}_{order}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_z_{Parameters['z']}_calibrated_{j}.p", "rb" ) )
        
        for i in range(1,n_pixels):
            zb = Parameters['z'] - i*dl*H(zb)/(c)
            ze = zb - dl*H(zb)/(c)
            tau_z[j] = tau_z[j] + np.abs((xh[i]*(den[i]+1))*(tau_gp*R_alpha/np.pi)*(((1+zb)/(1+z))**(3/2))*np.abs((np.abs(I((1+zb)/(1+z)))-np.abs(I((1+ze)/(1+z)))))) # Mesinger
    
        tau_z[j][0] = 10
        for k in range(n_pixels-1,0,-1):
            if(tau_z[j][k]>=5) | (tau_z[j][k] < 0):
                break

        for l in range(k,0,-1):
            tau_z[j][l] = 10 
    
        tau_avg = tau_avg + tau_z[j]/N
        
        e_tau_z = np.exp(-tau_z[j])
        lamda = (1+z)*1215.67/(1+Parameters['z'])

        plt.plot(lamda,e_tau_z)
    
    tau_scatter = (tau_z - tau_avg)**2

    
    e_tau_avg = np.exp(-tau_avg)
    
    pickle.dump(lamda,open( f"{newpath}/lamda_z_{Parameters['z']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_z_{Parameters['z']}_calibrated.p", "wb" ))
    pickle.dump(e_tau_avg,open(f"{newpath}/e_tau_avg_Mass_{base}_{order}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_z_{Parameters['z']}_calibrated.p","wb"))
    pickle.dump(tau_z,open(f"{newpath}/tau_z_all_Mass_{base}_{order}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_z_{Parameters['z']}_calibrated.p","wb"))
    
    plt.plot(lamda,e_tau_avg,'k--' , label = 'Average value')
    ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
    ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
    plt.xlim(1170,1270)
    plt.tight_layout()
    #legend = ax.legend(loc='center right', shadow=True, fontsize='large')
    #legend.get_frame().set_facecolor('C0')
    plt.savefig(f"{newpath}/Plots/Damping_wing_Mass_{base}_{order}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_z_{Parameters['z']}_calibrated.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    
    base = []
    order = []
    file = open(f"{newpath}/Halos_for_skewers.txt",'r')
    for l in file.readlines():
        b, o = l.strip().split(" ")
        base.append(int(b))
        order.append(int(o))
        
    base = np.array(base)
    order = np.array(order)    
    
    for i in range(0,len(base)):
        Damping_Wings(base[i], order[i])
