#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:44:34 2022

@author: sharma

cleaned up versions of the code
"""

#-----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import random
import pickle
import datetime
import os
import sys
import importlib

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
today = str(datetime.date.today())
#newpath = r'/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/'+today 
# if not os.path.exists(newpath):
#     print("Could not find the directory to load data")
#     sys.exit()


#-----------------------------------------------------------------------------
#New path for the code in the laptop
newpath = r'/Users/sharma/work/21cmFast_codes_and_plots/'+today 

if not os.path.exists(newpath):
    print("Could not find the directory to load data")
    sys.exit()

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
tq = 3.154*10**13 # units: s
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
#   Point on the surface of unit sphere

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

v = np.linspace(0, 1, 20)
phi = np.arccos(2*v-1)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

#-----------------------------------------------------------------------------
#Hubble Rate
def H(z):
    return H0*(Omega_m*(1+z)**3 + Omega_lambda + Omega_k*((1+z)**2))**(1/2)    
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def Calculate_skewers(rank,base_halo_mass,o_halo_mass,n_halos,new_halo_coords,new_halo_mass,ionised_box,density_field):    
    '''
    Description:
        Calculates the neutral fraction weighted over density from different halos along some random sightlines

    Parameters
    ----------
    rank : Integer
        To select the specific parameters file and use for parallelisation 
    base_halo_mass : Integer
        Base of the value of halo mass, it is provided to seperate halos and thier data files
    o_halo_mass : Integer
        Order of the value of halo mass, it is provided to seperate halos and thier data files
    n_halos : Integer
        Number of halos in a given mass bin
    new_halo_coords : Array
        x,y,z coordinates of the halos in the given mass bin
    new_halo_mass : Array
        masses of the halos in the given mass bin
    ionised_box : Array
        Neutral fraction at each pixels in the box
    density_field : Array
        Density of matter at each pixel in the box

    Returns
    -------
    None.

    '''
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Importing parameters 

    Para = importlib.import_module(f'Parameters_temp_{rank}')
    Parameters = Para.Parameters

    len_z = n_pixels
    z_red = np.linspace(Parameters['z'],Parameters['z']-1.0,num=len_z)
    delta_z = 1/(n_pixels-1)

    #-----------------------------------------------------------------------------
    
    # Storing x,y and z coords of the halos seperately
    
    halo_x_coord = new_halo_coords[:,0]*Parameters['BOX_LEN']/Parameters['HII_DIM'] #Coords of halos
    halo_y_coord = new_halo_coords[:,1]*Parameters['BOX_LEN']/Parameters['HII_DIM']
    halo_z_coord = new_halo_coords[:,2]*Parameters['BOX_LEN']/Parameters['HII_DIM']
    
    
    xx = np.linspace(0,Parameters['HII_DIM']-1,Parameters['HII_DIM'])   #Region of x coordinates
    yy = np.linspace(0,Parameters['HII_DIM']-1,Parameters['HII_DIM'])   #Region of y coordinates 
    zz = np.linspace(0,Parameters['HII_DIM']-1,Parameters['HII_DIM'])   #Region of z coordinates
    pixels = np.linspace(0,n_pixels-1,n_pixels)
    
    interp_ionised_box = RegularGridInterpolator((xx,yy,zz), ionised_box)   #Interpolation function to calculate inonisation field
    interp_density_field = RegularGridInterpolator((xx,yy,zz), density_field)   #Interpolation function to calculate density field
    
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    #Calculting the skewers
    
    dr = np.arange(0,n_pixels,delta)    # Distance travelled across any vector
    
    # Storing the coordinates along the sightline for each halo
    
    X = [[]*n_pixels]*100
    Y = [[]*n_pixels]*100
    Z = [[]*n_pixels]*100   
    
    # Tolerence value for ionized sphere around a halo
    tol = 0.5
    Rion = 0.0
           
    Random_halo = np.zeros(100)
    for i in range(0,100):
        Random_halo[i] = random.randrange(len(halo_x_coord))
    #ßprint(Random_halo)
    for i in range(0,100):     #for i in range(0,halo_10_11.sum()): #Loop over all sightlines
        #Updating coords along a random direction
        xi, yi, zi = sample_spherical(1)    #random direction vector
        #print(sample_spherical(1))
        
        X[i] = halo_x_coord[int(Random_halo[i])] + dr*xi
        Y[i] = halo_y_coord[int(Random_halo[i])] + dr*yi
        Z[i] = halo_z_coord[int(Random_halo[i])] + dr*zi
        
        
        #Enabling the periodic boundary condition
        X[i] = X[i] - (Parameters['HII_DIM']-1)*(np.floor(X[i])//(Parameters['HII_DIM']-1))
        Y[i] = Y[i] - (Parameters['HII_DIM']-1)*(np.floor(Y[i])//(Parameters['HII_DIM']-1))
        Z[i] = Z[i] - (Parameters['HII_DIM']-1)*(np.floor(Z[i])//(Parameters['HII_DIM']-1))
        
        xh = np.zeros(n_pixels) # Calculating the x_h along (X,Y,Z)
        den = np.zeros(n_pixels) # Calculating the density along (X,Y,Z)
        Sum = np.zeros(n_pixels)
        r = np.zeros(n_pixels)
        
        for k in range(0,n_pixels-1):
            del_r = -(z_red[k+1] - z_red[k])*c/(H(z_red[k])*(1+z_red[k]))
            r[k+1] = r[k] + del_r
            XH = interp_ionised_box([X[i][k],Y[i][k],Z[i][k]])
            DEN = interp_density_field([X[i][k],Y[i][k],Z[i][k]])
            
            rho_c = (3*H(z_red[k])**2)*Conversion_amu_Mpc/(8*np.pi*G)
            n_HI = Omega_b*rho_c*XH*(DEN+1)
            Sum[k+1] = Sum[k] + n_HI*4*np.pi*r[k+1]*r[k+1]*del_r
        
        for l in range(n_pixels-1,0,-1):        
            if(np.abs(Sum[l] - Nion*tq)/(Nion*tq)<=tol):
                Rion = r[l]
                break

        for j in range (0,n_pixels):
            if(dr[j]<=Rion*(1+z_red[j])):    #add (1+z)
                xh[j] = 0.0
            else:
                xh[j] = interp_ionised_box([X[i][j],Y[i][j],Z[i][j]])
            den[j] = interp_density_field([X[i][j],Y[i][j],Z[i][j]])
    
        pickle.dump(xh,open( f"{newpath}/xh_HM_{base_halo_mass}_{o_halo_mass}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_{i}_no_halofield.p", "wb" ))
        pickle.dump(den,open( f"{newpath}/density_HM_{base_halo_mass}_{o_halo_mass}_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_{i}_no_halofield.p", "wb" ))
        
    plt.imshow(ionised_box[0],extent=[Parameters['BOX_LEN'] ,0,Parameters['BOX_LEN'] ,0], origin='upper')
    
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
    

if __name__ == '__main__':
    
    zzz=1
    
    # halo_mass = pickle.load(open(f"{newpath}/Halo_masses_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
    # halo_coords = pickle.load(open(f"{newpath}/Halo_coords_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p","rb"))
    # halo_mass_bins = np.unique(halo_mass)   #Checking the bins of halo masses
    # ionised_box = pickle.load( open(f"{newpath}/Ionized_box_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))
    # density_field = pickle.load( open(f"{newpath}/Density_field_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.p", "rb" ))

    # Mass_bins = np.unique(halo_mass)
    # n_Mass_bins = len(Mass_bins)
    
    # file = open(f"{newpath}/Halos_for_skewers_T_vir_{Parameters['T_vir']}_M_Turn_{Parameters['M_min']}_target_xh_{Parameters['target_xh']}_alpha_esc_{Parameters['alpha_esc']}_alpha_star_{Parameters['alpha_star']}_f_star_{Parameters['f_star']}_z_{Parameters['z']}_calibrated_no_halofield.txt", 'w')
    
    # for i in range(0,n_Mass_bins,int(n_Mass_bins/5)):
    #     m = (halo_mass == Mass_bins[i])
    #     new_halo_mass = halo_mass[m]
    #     new_halo_coords = halo_coords[m]
    #     n_halos = len(new_halo_mass)
    #     o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
    #     base_halo_mass = int(np.round(new_halo_mass[1]/(10**o_halo_mass),0))
    #     print(base_halo_mass,o_halo_mass)
        
    #     file.write(f"{base_halo_mass} {o_halo_mass} \n")
    #     # file.write("\t")
    #     # file.write(str(o_halo_mass))
    #     # file.write('\n')
    #     #pickle.dump({base_halo_mass,o_halo_mass}, open(f"{newpath}/Halos_for_skewers","wb"))
        
    #     Calculate_skewers(base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)
    
    # m = (halo_mass == Mass_bins[n_Mass_bins-1])
    # new_halo_mass = halo_mass[m]
    # new_halo_coords = halo_coords[m]
    # n_halos = len(new_halo_mass)
    # o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))
    # base_halo_mass = int(np.round(new_halo_mass[0]/(10**o_halo_mass),0))
    # print(base_halo_mass,o_halo_mass)
    # file.write(f"{base_halo_mass} {o_halo_mass}")
    # file.close()
    # #pickle.dump({base_halo_mass,o_halo_mass}, open(f"{newpath}/Halos_for_skewers","wb"))
    
    # Calculate_skewers(base_halo_mass, o_halo_mass, n_halos, new_halo_coords, new_halo_mass, ionised_box, density_field)
    
