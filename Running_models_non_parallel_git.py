#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:20:11 2022

@author: sharma
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import pickle
import datetime
import os
import sys
#from mpi4py import MPI

#-----------------------------------------------------------------------------
#For cluster
today = str(datetime.date.today())
# newpath = r'/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/'+today 
# if not os.path.exists(newpath):
#     os.mkdir(newpath)

# plotpath = newpath+'/Plots'
# if not os.path.exists(plotpath):
#     os.mkdir(plotpath)
    
#For laptop
newpath = r'/Users/sharma/work/21cmFast_codes_and_plots/'+today 
if not os.path.exists(newpath):
    os.mkdir(newpath)

plotpath = newpath+'/Plots'
if not os.path.exists(plotpath):
    os.mkdir(plotpath)
    
#-----------------------------------------------------------------------------

# For parallelization

# if not os.path.exists('Parameters_temp.py'):

#     import Parameters_file as Params
    
#     #target_xh = 0.6
#     #z = 9
#     variables = {'target_xh': 0.5, 'z':7}
#     #f = open(f'Parameters_temp_{rank}.py', 'w')
#     f = open(f'/Users/sharma/work/21cmFast_codes_and_plots/Parameters_temp.py', 'w')
#     f.write('Parameters = {\n')
    
#     for key, value in variables.items():
#         f.write(f"\t'{key}':{variables[key]},\n")
                
#     for p in Params.Parameters:
#         if (p not in variables):
#             f.write(f"\t'{p}':{Params.Parameters[p]},\n")
#             #print(p)
#     f.seek(0,2)
#     f.seek(f.tell() -2,0)
#     f.truncate()
#     f.write('}')
#     f.close()
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

# Note: Parameters that we are playing with are : redshift 'z', global mean neutral fraction 'xh' (calibrated by f_esc), alpha star, alpha escape, f star, Turn over mass 'M_min'

'''
rough sketch : the bounds on the parameters are taken from Greig 2019:
    alpha star : 0 to 1
    alpha escape : -1 to 0
    xh = 0.1 to 0.9
    f star = -0.25 to -2
    z = 6 to 14
    M_min  = 8 to 9.5
    
    Parameters = {
        'z':7.0,
        'M_min':9,
        'T_vir':5,
        'target_xh':0.75,
        'alpha_esc':0.2,
        'alpha_star': 0,
        'f_star':-0.25,
        'DIM':1024,
        'HII_DIM':256,
        'BOX_LEN':200
        }
'''

z_list = np.linspace(6,14,2)
M_min_list = np.linspace(8,9.5,2)
target_xh_list = np.linspace(0.1,0.9,2)
alpha_esc_list = np.linspace(-1,0,2)
alpha_star_list = np.linspace(0,1,2)
f_star_list = np.linspace(-2,-0.25,2)

import Parameters_file as Params

variables = {'target_xh': 0.25, 'z':7}

#if not os.path.exists('Parameters_temp.py'):

f = open(f'/Users/sharma/work/21cmFast_codes_and_plots/Parameters_temp.py', 'w')
f.write('Parameters = {\n')

for key, value in variables.items():
    f.write(f"\t'{key}':{variables[key]},\n")
            
for p in Params.Parameters:
    if (p not in variables):
        f.write(f"\t'{p}':{Params.Parameters[p]},\n")
        #print(p)
f.seek(0,2)
f.seek(f.tell() -2,0)
f.truncate()
f.write('}')
f.close()

#-----------------------------------------------------------------------------
import Generating_ionized_boxes_git as GIB
import Calculating_skewers_git as CS
import Damping_wings_git as DW

from Parameters_temp import Parameters

print(Parameters)
#-----------------------------------------------------------------------------

GIB.Generate_ion_boxes(newpath)

halo_mass = pickle.load(open(f"{newpath}/Halo_masses_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
halo_coords = pickle.load(open(f"{newpath}/Halo_coords_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
halo_mass_bins = np.unique(halo_mass)   #Checking the bins of halo masses
ionised_box = pickle.load( open(f"{newpath}/Ionized_box_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))
density_field = pickle.load( open(f"{newpath}/Density_field_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

Mass_bins = np.unique(halo_mass)
n_Mass_bins = len(Mass_bins)


file = open(f"{newpath}/Halos_for_skewers.txt", 'w')

for i in range(0,n_Mass_bins,int(n_Mass_bins/5)):
    m = (halo_mass == Mass_bins[i])
    new_halo_mass = halo_mass[m]
    new_halo_coords = halo_coords[m]
    n_halos = len(new_halo_mass)
    o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
    base_halo_mass = int(np.round(new_halo_mass[1]/(10**o_halo_mass),0))
    print(base_halo_mass,o_halo_mass)
    
    file.write(f"{base_halo_mass} {o_halo_mass} \n")
    # file.write("\t")
    # file.write(str(o_halo_mass))
    # file.write('\n')
    #pickle.dump({base_halo_mass,o_halo_mass}, open(f"{newpath}/Halos_for_skewers","wb"))
    
    CS.Calculate_skewers(base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)

m = (halo_mass == Mass_bins[n_Mass_bins-1])
new_halo_mass = halo_mass[m]
new_halo_coords = halo_coords[m]
n_halos = len(new_halo_mass)
o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
base_halo_mass = int(np.round(new_halo_mass[0]/(10**o_halo_mass),0))
print(base_halo_mass,o_halo_mass)
file.write(f"{base_halo_mass} {o_halo_mass}")
file.close()
#pickle.dump({base_halo_mass,o_halo_mass}, open(f"{newpath}/Halos_for_skewers","wb"))

CS.Calculate_skewers(base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)

base = []
order = []
file = open(f"{newpath}/Halos_for_skewers.txt",'r')
for l in file.readlines():
    b, o = l.strip().split(" ")
    base.append(int(b))
    order.append(int(o))
    
base = np.array(base)
order = np.array(order)    
#sys.exit()

for i in range(0,len(base)):
    DW.Damping_Wings(base[i], order[i])
 
#Plottting the damping wings

lamda = pickle.load(open(f"{newpath}/lamda_z_{Parameters['z']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

e_tau_avg = []
tau_z = []

plt.figure(figsize=(5, 6), dpi=150)
ax = plt.axes()
plt.minorticks_on() 
ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
plt.title(f"Averaged damping wings for various halo masses (xh = {Parameters['target_xh']})")
for i in range(0,len(base)):
    e_tau_avg.append(pickle.load(open(f"{newpath}/e_tau_avg_Mass_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
    tau_z.append(pickle.load(open(f"{newpath}/tau_z_all_Mass_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
    plt.plot(lamda,e_tau_avg[i], label = f'{base[i]}*{order[i]}')

ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
plt.xlim(1200,1240)
plt.tight_layout()
legend = plt.legend(loc='upper left', fontsize='small')
plt.savefig(f"{plotpath}/Damping_wings_averaged_xh_{Parameters['target_xh']}_no_halofield.png")
plt.show()
plt.close()
#-----------------------------------------------------------------------------
