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
import importlib
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
    
#Constants

n_pixels = 300
delta = 1 #differential interval to move in pixel space

H0 = 70000.0    # units: m/s/Mpc
Omega_m = 0.3
Omega_lambda = 0.7
Omega_k = 0.0
Omega_b = 0.045
c = 3*10**8 #ms^-1

G = 6.67*10**(-11) # units: m*((m/s)**2)/kg
Conversion_amu_Mpc = (6.022/3.241)*10**49
Nion = 10**57
tq = 3.154*10**13
h = 0.7
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------

# Note: Parameters that we are playing with are : redshift 'z', global mean neutral fraction 'xh' (calibrated by f_esc), alpha star, alpha escape, f star, Turn over mass 'M_min'

'''
rough sketch : the bounds on the parameters are taken from Greig 2019:
    
    z = 6 to 12
    M_min  = Mp*20 to Mp*200
    xh = 0.1 to 0.9
    alpha escape : -1 to 0
    alpha star : 0 to 1
    f star = -0.25 to -2
    
    Parameters = {
        'z':7.0,
        'M_min':9,
        'T_vir':5,
        'target_xh':0.75,
        'alpha_esc':-0.5,
        'alpha_star': 0.5,
        'f_star':-0.225,
        'DIM':1024,
        'HII_DIM':256,
        'BOX_LEN':200
        }
'''

z_list = np.linspace(6,7,2)
M_min_list = np.linspace(9.3,10.3,2)
target_xh_list = np.linspace(0.1,0.9,2)
alpha_esc_list = np.linspace(-1,0,2)
alpha_star_list = np.linspace(0,1,2)
f_star_list = np.linspace(-2,-0.25,2)

n_pixels = 300

import Calculating_M_pixels as MP
import Parameters_file as Params

M_p = MP.Get_me_M_min(Params.Parameters['BOX_LEN'],Params.Parameters['DIM'])

M_min = np.log10(M_p*20)

'''
# Running all models

for I in range(0,2):
    for J in range(0,2):
        for K in range(0,2):
            for L in range(0,2):
                for M in range(0,2):
                    Omega_m_z = (Omega_m*(1+I)**3)/(Omega_m*(1+I)**3 + Omega_lambda)
                    d = Omega_m_z**2 -1
                    Delta_c = 18*np.pi**2 +82*d -39*d**2
                    mu = J*0.5 + (1-J)

                    T_vir = (1.98*10**4)*(mu/0.6)*((M_p*20*h/10**8)**(2/3))*(Omega_m*Delta_c/(Omega_m_z*18*np.pi**2))*((1+I)/10)
                    
                    variables = {
                        'z':z_list[I],
                        'M_min':M_min,
                        'T_vir':np.log10(T_vir),
                        'target_xh':target_xh_list[J],
                        'alpha_esc':alpha_esc_list[K],
                        'alpha_star': alpha_star_list[L],
                        'f_star':f_star_list[M]
                        }
                    #print(variables)
                    
                    rank = 1*M + 2*L + 4*K + 8*J + 16*I
                    
                    #print(rank)

                    f = open(f'/Users/sharma/work/21cmFast_codes_and_plots/Parameters_temp_{rank}.py', 'w')
                    f.write('Parameters = {\n')

                    for key, value in variables.items():
                        f.write(f"\t'{key}':{variables[key]},\n")
                                
                    for p in Params.Parameters:
                        if (p not in variables):
                            f.write(f"\t'{p}':{Params.Parameters[p]},\n")
                    f.seek(0,2)
                    f.seek(f.tell() -2,0)
                    f.truncate()
                    f.write('}')
                    f.close()

                    #-----------------------------------------------------------------------------
                    Para = importlib.import_module(f'Parameters_temp_{rank}')
                    Parameters = Para.Parameters
                    #from Parameters_temp import Parameters

                    print(Parameters)
                    #-----------------------------------------------------------------------------
                    
                    #-----------------------------------------------------------------------------
                    import Generating_ionized_boxes_git as GIB
                    import Calculating_skewers_git as CS
                    import Damping_wings_git as DW
                    #-----------------------------------------------------------------------------


                    #-----------------------------------------------------------------------------

                    GIB.Generate_ion_boxes(newpath,rank)

                    halo_mass = pickle.load(open(f"{newpath}/Halo_masses_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
                    halo_coords = pickle.load(open(f"{newpath}/Halo_coords_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
                    halo_mass_bins = np.unique(halo_mass)   #Checking the bins of halo masses
                    ionised_box = pickle.load( open(f"{newpath}/Ionized_box_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))
                    density_field = pickle.load( open(f"{newpath}/Density_field_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

                    Mass_bins = np.unique(halo_mass)
                    n_Mass_bins = len(Mass_bins)


                    file = open(f"{newpath}/Halos_for_skewers_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.txt", 'w')

                    for i in range(0,n_Mass_bins,int(n_Mass_bins/5)):
                        m = (halo_mass == Mass_bins[i])
                        new_halo_mass = halo_mass[m]
                        new_halo_coords = halo_coords[m]
                        n_halos = len(new_halo_mass)
                        o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
                        base_halo_mass = int(np.round(new_halo_mass[1]/(10**o_halo_mass),0))
                        print(base_halo_mass,o_halo_mass, n_halos, np.log10(new_halo_mass[0]))
                        
                        file.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
                        # file.write("\t")
                        # file.write(str(o_halo_mass))
                        # file.write('\n')
                        #pickle.dump({base_halo_mass,o_halo_mass}, open(f"{newpath}/Halos_for_skewers","wb"))
                        
                        CS.Calculate_skewers(rank,base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)

                    if bool((n_Mass_bins-1)%5):
                        m = (halo_mass == Mass_bins[n_Mass_bins-1])
                        new_halo_mass = halo_mass[m]
                        new_halo_coords = halo_coords[m]
                        n_halos = len(new_halo_mass)
                        o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
                        base_halo_mass = int(np.round(new_halo_mass[0]/(10**o_halo_mass),0))
                        print(base_halo_mass,o_halo_mass, n_halos, np.log10(new_halo_mass[0]))
                        file.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}s")

                    file.close()
                    #pickle.dump({base_halo_mass,o_halo_mass}, open(f"{newpath}/Halos_for_skewers","wb"))

                    CS.Calculate_skewers(rank,base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)

                    base = []
                    order = []
                    num_halos = []
                    mass_halos = []
                    file = open(f"{newpath}/Halos_for_skewers_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.txt",'r')
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
                        DW.Damping_Wings(rank,base[i], order[i])
                     
                    #Plottting the damping wings

                    lamda = pickle.load(open(f"{newpath}/lamda_z_{Parameters['z']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

                    e_tau_avg = np.zeros((len(base),n_pixels))
                    low_quantile = np.zeros((len(base),n_pixels))
                    mid_quantile = np.zeros((len(base),n_pixels))
                    up_quantile = np.zeros((len(base),n_pixels))

                    plt.figure(figsize=(5, 6), dpi=150)
                    ax = plt.axes()
                    plt.minorticks_on() 
                    ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
                    ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
                    #plt.title(f"Averaged damping wings for various halo masses (xh = {Parameters['target_xh']})")

                    for i in range(0,len(base)):
                        e_tau_avg[i] = (pickle.load(open(f"{newpath}/e_tau_avg_Mass_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
                        low_quantile[i] = (pickle.load(open(f"{newpath}/lower_quantile_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
                        mid_quantile[i] = (pickle.load(open(f"{newpath}/middle_quantile_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
                        up_quantile[i] = (pickle.load(open(f"{newpath}/upper_quantile_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
                        #tau_z[i] = (pickle.load(open(f"{newpath}/tau_z_all_Mass_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
                        plt.plot(lamda,e_tau_avg[i], label = f'({mass_halos[i]}, {num_halos[i]})')
                        # plt.plot(lamda,low_quantile[i], 'k--')
                        # plt.plot(lamda,up_quantile[i], 'k--')
                        plt.fill_between(lamda, up_quantile[i], low_quantile[i], alpha=0.2)

                    ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
                    ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
                    plt.xlim(1170,1240)
                    plt.tight_layout()
                    legend = plt.legend(loc='lower right', fontsize='small')
                    plt.savefig(f"{plotpath}/Damping_wings_averaged_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png")
                    plt.show()
                    plt.close()


                    plt.figure(figsize=(5, 6), dpi=150)
                    ax = plt.axes()
                    plt.minorticks_on() 
                    ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
                    ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)

                    plt.plot(lamda,mid_quantile[0], label = f'({mass_halos[0]}, {num_halos[0]})',)
                    plt.fill_between(lamda, up_quantile[0], low_quantile[0], alpha=0.25)

                    plt.plot(lamda,mid_quantile[len(base)-1], label = f'({mass_halos[5]}, {num_halos[5]})',)
                    plt.fill_between(lamda, up_quantile[len(base)-1], low_quantile[len(base)-1], alpha=0.25)

                    plt.axvline(x = 1215.67, ymin=0.0, ymax=1.0, color = 'black', linestyle='dashed', label = r"$Ly_{\alpha}$")

                    ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
                    ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
                    plt.xlim(1170,1270)
                    plt.tight_layout()
                    legend = plt.legend(loc='center right', fontsize='small')
                    plt.savefig(f"{plotpath}/quantile_plot_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png")
                    plt.show()
                    plt.close()


                    #-----------------------------------------------------------------------------
'''

#sys.exit()




#-----------------------------------------------------------------------------
#Middle model


Z = 6
xh = 0.5

Omega_m_z = (Omega_m*(1+Z)**3)/(Omega_m*(1+Z)**3 + Omega_lambda)
d = Omega_m_z**2 -1
Delta_c = 18*np.pi**2 +82*d -39*d**2
mu = xh*0.5 + (1-xh)

T_vir = (1.98*10**4)*(mu/0.6)*((M_p*20*h/10**8)**(2/3))*(Omega_m*Delta_c/(Omega_m_z*18*np.pi**2))*((1+Z)/10)



variables = {
    'z':Z,
    'M_min':M_min,
    'T_vir':np.log10(T_vir),
    'target_xh':xh,
    'alpha_esc':-0.5,
    'alpha_star': 0.5,
    'f_star':-1.125
    }


#print(variables)

rank = 999

#print(rank

f = open(f'/Users/sharma/work/21cmFast_codes_and_plots/Parameters_temp_{rank}.py', 'w')
f.write('Parameters = {\n')

for key, value in variables.items():
    f.write(f"\t'{key}':{variables[key]},\n")
            
for p in Params.Parameters:
    if (p not in variables):
        f.write(f"\t'{p}':{Params.Parameters[p]},\n")
f.seek(0,2)
f.seek(f.tell() -2,0)
f.truncate()
f.write('}')
f.close()

#-----------------------------------------------------------------------------
Para = importlib.import_module(f'Parameters_temp_{rank}')
Parameters = Para.Parameters
#from Parameters_temp import Parameters

print(Parameters)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
import Generating_ionized_boxes_git as GIB
import Calculating_skewers_git as CS
import Damping_wings_git as DW
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------

GIB.Generate_ion_boxes(newpath,rank)

halo_mass = pickle.load(open(f"{newpath}/Halo_masses_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
halo_coords = pickle.load(open(f"{newpath}/Halo_coords_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
halo_mass_bins = np.unique(halo_mass)   #Checking the bins of halo masses
ionised_box = pickle.load( open(f"{newpath}/Ionized_box_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))
density_field = pickle.load( open(f"{newpath}/Density_field_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

Mass_bins = np.unique(halo_mass)
n_Mass_bins = len(Mass_bins)


file = open(f"{newpath}/Halos_for_skewers_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.txt", 'w')

for i in range(0,n_Mass_bins,int(n_Mass_bins/5)):
    m = (halo_mass == Mass_bins[i])
    new_halo_mass = halo_mass[m]
    new_halo_coords = halo_coords[m]
    n_halos = len(new_halo_mass)
    o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
    base_halo_mass = int(np.round(new_halo_mass[1]/(10**o_halo_mass),0))
    print(base_halo_mass,o_halo_mass, n_halos, np.log10(new_halo_mass[0]))
    
    file.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
    # file.write("\t")
    # file.write(str(o_halo_mass))
    # file.write('\n')
    #pickle.dump({base_halo_mass,o_halo_mass}, open(f"{newpath}/Halos_for_skewers","wb"))
    
    CS.Calculate_skewers(rank,base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)

if bool((n_Mass_bins-1)%5):
    m = (halo_mass == Mass_bins[n_Mass_bins-1])
    new_halo_mass = halo_mass[m]
    new_halo_coords = halo_coords[m]
    n_halos = len(new_halo_mass)
    o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
    base_halo_mass = int(np.round(new_halo_mass[0]/(10**o_halo_mass),0))
    print(base_halo_mass,o_halo_mass, n_halos, np.log10(new_halo_mass[0]))
    file.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}s")

file.close()
#pickle.dump({base_halo_mass,o_halo_mass}, open(f"{newpath}/Halos_for_skewers","wb"))

CS.Calculate_skewers(rank,base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)

base = []
order = []
num_halos = []
mass_halos = []
file = open(f"{newpath}/Halos_for_skewers_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.txt",'r')
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
    DW.Damping_Wings(rank,base[i], order[i])
 
#Plottting the damping wings

lamda = pickle.load(open(f"{newpath}/lamda_z_{Parameters['z']}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

e_tau_avg = np.zeros((len(base),n_pixels))
low_quantile = np.zeros((len(base),n_pixels))
mid_quantile = np.zeros((len(base),n_pixels))
up_quantile = np.zeros((len(base),n_pixels))

plt.figure(figsize=(5, 6), dpi=150)
ax = plt.axes()
plt.minorticks_on() 
ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)
#plt.title(f"Averaged damping wings for various halo masses (xh = {Parameters['target_xh']})")

for i in range(0,len(base)):
    e_tau_avg[i] = (pickle.load(open(f"{newpath}/e_tau_avg_Mass_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
    low_quantile[i] = (pickle.load(open(f"{newpath}/lower_quantile_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
    mid_quantile[i] = (pickle.load(open(f"{newpath}/middle_quantile_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
    up_quantile[i] = (pickle.load(open(f"{newpath}/upper_quantile_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
    #tau_z[i] = (pickle.load(open(f"{newpath}/tau_z_all_Mass_{base[i]}_{order[i]}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" )))
    plt.plot(lamda,e_tau_avg[i], label = f'({mass_halos[i]}, {num_halos[i]})')
    # plt.plot(lamda,low_quantile[i], 'k--')
    # plt.plot(lamda,up_quantile[i], 'k--')
    plt.fill_between(lamda, up_quantile[i], low_quantile[i], alpha=0.2)

ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
plt.xlim(1170,1240)
plt.tight_layout()
legend = plt.legend(loc='lower right', fontsize='small')
plt.savefig(f"{plotpath}/Damping_wings_averaged_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png")
plt.show()
plt.close()


plt.figure(figsize=(5, 6), dpi=150)
ax = plt.axes()
plt.minorticks_on() 
ax.tick_params('both', which='major', length=16, width=1, direction='in', top=True, right=True)
ax.tick_params('both', which='minor', length=8, width=1, direction='in', top=True, right=True)

plt.plot(lamda,mid_quantile[0], label = f'({mass_halos[0]}, {num_halos[0]})',)
plt.fill_between(lamda, up_quantile[0], low_quantile[0], alpha=0.25)

plt.plot(lamda,mid_quantile[len(base)-1], label = f'({mass_halos[5]}, {num_halos[5]})',)
plt.fill_between(lamda, up_quantile[len(base)-1], low_quantile[len(base)-1], alpha=0.25)

plt.axvline(x = 1215.67, ymin=0.0, ymax=1.0, color = 'black', linestyle='dashed', label = r"$Ly_{\alpha}$")

ax.set_ylabel(r'$e^{-\tau{D}}$',  fontsize=20)
ax.set_xlabel(r"$\lambda_{\alpha}(1+z)~~Å$",  fontsize=20)
plt.xlim(1170,1270)
plt.tight_layout()
legend = plt.legend(loc='center right', fontsize='small')
plt.savefig(f"{plotpath}/quantile_plot_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.png")
plt.show()
plt.close()
