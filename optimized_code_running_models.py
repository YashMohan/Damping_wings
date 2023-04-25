#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:12:17 2023

@author: sharma
"""
import numpy as np
import sys
#from mpi4py import MPI
import logging
from ordered_set import OrderedSet
import itertools
import pickle
from tqdm import tqdm
from multiprocessing import Process
from tabulate import tabulate

#-----------------------------------------------------------------------------
# Models
# Constants
from constants import *
# Parameters file
import parameters_file as Params
import generating_ionized_boxes_git as GIB      # This code generates ionized boxes of the given parameters and initial conditions
import calculating_skewers_git as CS        # This code calculates the neutral fraction weighted over density from different halos along some random sightlines for a given ionized box
import damping_wings_git as DW         # For a given sightline, it calculates the damping wing profile for a specific halo mass host of a quasar
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

def Len(p):
    if p is None:
        return 0
    
    return len(p)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def calculate_T_vir (z,xh,M):
    Omega_m_z = (Omega_m*(1+z)**3)/(Omega_m*(1+z)**3 + Omega_lambda)                                 
    d = Omega_m_z**2 -1
    Delta_c = 18*np.pi**2 +82*d -39*d**2
    mu = xh*0.5 + (1-xh)
    T_vir = (1.98*10**4)*(mu/0.6)*((10**(M)*h/10**8)**(2/3))*(Omega_m*Delta_c/(Omega_m_z*18*np.pi**2))*((1+z)/10)
    T_vir =  float('{:.2f}'.format(np.log10(T_vir)))
    return T_vir

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
class Models():
    '''
    DD
    Parameters_names
    Parameters_values
    Parameters_lens
    rank
    rank_list
    P_va, n_va
    P_vo, n_vo
    '''
    
    def __init__(self, Param_Ranges):
    
        self.DD = len(Params.Parameters)    # Total number of parameters we have, -1 since we are not varying redshift
        self.Parameters_names = [k for k in Param_Ranges]    # Storing the names of the parameters
        self.Parameters_names = [k.replace('_list','') for k in self.Parameters_names] # removing _list from the Parameters names
        self.Parameters_values = [v for v in Param_Ranges.values()]       # Storing the range values in Parameters_values 
        self.Parameters_values = [np.insert(self.Parameters_values[index],0,Params.Parameters[f'{keys}']) for index,keys in enumerate(Params.Parameters)]    # Adding the fiducial values at front
        self.Parameters_values = [list(OrderedSet(i)) for i in self.Parameters_values]     # Making sure that the fiducial value appeaers only at front, hence ignoring if it repeats somewhere
        self.Parameters_lens = [len(i) for i in self.Parameters_values]   # length of each parameter, i.e., number of points for each parameter
        self.Varying_order = [np.prod(self.Parameters_lens[self.DD-1:i:-1]) for i in range(0,self.DD-1)] + [0]  # The point after which each parameter start to vary for the first time
        self.rank_list = list(itertools.product(*self.Parameters_values))   # List of all possible combinations of parameter values
    
    def rank_calculation(self, Specific_values = None, P_VA = None, P_VO = None):
        '''
        add some tests
        
        Parameters
        ----------
        Specific_parameters : TYPE, optional
        DESCRIPTION. The default is None.
        P_VA : TYPE, optional
        DESCRIPTION. The default is None.
        P_VO : TYPE, optional
        DESCRIPTION. The default is None.
        
        Returns
        -------
        TYPE
        DESCRIPTION.
        
        '''
        self.P_va = None   # This is to make sure, even if the same object calls rank_calculation multiple times, each instances do not overlap
        self.P_vo = None
        self.n_va = None
        self.n_vo = None
        
        if P_VA is not None and P_VO is not None:
            if any(P for P in P_VA if P in P_VO):
                print('P_VA and P_VO cannot be the same')
            
                return None
        
        if Specific_values is not None:
            Paramms = dict(Params.Parameters)  # Using rest of the values from fiducial model
            Paramms.update(Specific_values)   # Updating the specific values on fiducial model
            indicies = [self.Parameters_values[i].index(Paramms[keys]) for i,keys in enumerate(Paramms)]  # Picking up the indicies of each parameter to calculate the rank corresponding to the specific parameters combo
            self.rank = [sum(j*self.Varying_order[i] for i,j in enumerate(indicies))]   # Getting the rank for the specific parameters combo
        
            return self.rank
        
        if P_VA is not None:
            P_VA = set(P_VA)
            self.P_va = [self.Parameters_names.index(j) for j in P_VA]  # If P_VA is provided, then storing the position of the provided parameters in P_va
            self.n_va = [np.linspace(0,self.Parameters_lens[i]-1,self.Parameters_lens[i], dtype=int) for i in self.P_va] # Storing the length or number of points for P_VA parameter list in n_va
        
        if P_VO is not None:
            self.P_vo = [self.Parameters_names.index(j) for j in P_VO]  # If P_VO is provided, then storing the position of the provided parameters in P_vo
            self.n_vo = [np.linspace(0,self.Parameters_lens[i]-1,self.Parameters_lens[i], dtype=int) for i in self.P_vo]  # Storing the length or number of points for P_VO parameter list in n_vo
        
        if Len(self.P_vo) == 1 and self.P_va is not None:  # If there's only one parameter in P_vo, then it can be clubbed with P_va
            self.P_va.extend(self.P_vo)
            self.n_va.extend(self.n_vo)
            self.P_vo = None
            self.n_vo = None
        
        ranks_a = [0]
        if self.P_va is not None:
            P_vary_a = [(self.Varying_order[j]*self.n_va[i]) for i,j in enumerate(self.P_va)]
            res_a = list(itertools.product(*P_vary_a))
            ranks_a = list(map(sum, res_a))
        
        P_vary_o = [0]
        if self.P_vo is not None:
            P_vary_o = np.concatenate([self.n_vo[i]*self.Varying_order[j] for i,j in enumerate(self.P_vo)], axis=0)
            P_vary_o = np.unique(P_vary_o).tolist()
        res_o = list(itertools.product(P_vary_o,ranks_a))
        self.rank = list(map(sum, res_o))
        self.rank.sort()
        
        return self.rank
    
    def halo_data(self,i):
        '''
        

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.

        Returns
        -------
        new_halo_mass : TYPE
            DESCRIPTION.
        new_halo_coords : TYPE
            DESCRIPTION.
        n_halos : TYPE
            DESCRIPTION.
        o_halo_mass : TYPE
            DESCRIPTION.
        base_halo_mass : TYPE
            DESCRIPTION.

        '''
        Mass_bins = np.unique(self.halo_mass)
        m = (self.halo_mass == Mass_bins[i])     # Selecting only the halos of the desired mass
        new_halo_mass = self.halo_mass[m]        # Storing all those halos as new_halo_mass
        new_halo_coords = self.halo_coords[m]    # Storing their corresponding coordinates in new_halo_coords
        n_halos = len(new_halo_mass)        # Number of halos in the selected mass bin
        o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))             # The order of the mass of the halo in the form of Mass = base*10^order, this is an approximation not the exact value of the masses
        base_halo_mass = int(np.round(new_halo_mass[0]/(10**o_halo_mass),0))        # The base from Mass = base*10^order, base and order are used to seperate halos of different masses                
        
        return new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass 
    
    def damping_wings_calculations(self, i, Parameters, I):
        '''
        

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.
        Parameters : TYPE
            DESCRIPTION.
        I : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass = self.halo_data(i)
        
        # Calling the calculate skewer function
        CS.Calculate_skewers(base_halo_mass, o_halo_mass, new_halo_coords, self.ionised_box, self.density_field, Parameters, I)
        DW.Damping_Wings(base_halo_mass, o_halo_mass, Parameters, I)  # Calculating damping wings for different halo mass

    
    def modelling(self, initial_conditions, rank = None):
        '''
        
        
        Parameters
        ----------
        rank : TYPE, optional
        DESCRIPTION. The default is None.
        
        Returns
        -------
        None.
        
        '''
    
        if rank is None:
            rank = [0]
            
        self.rank = rank
        
        for I in self.rank:
        
            # Calling the Generate Ionized box function 
            Parameters = dict(zip(self.Parameters_names,self.rank_list[I]))
            Parameters['T_vir'] = calculate_T_vir(z = Parameters['z'], xh = Parameters['target_xh'], M = Parameters['M_min'])
            print(Parameters)
            GIB.Generate_ion_boxes(initial_conditions,Parameters,I)   # Generating the ionized box
            
            #--------------------------------------------------------------------------------------------------------------------------------------------------------
            # Skewers calculations
            print("\nLoading the halo files and the ionized box")
            # Loading all the halos with their masses and coords, and the ionized and desnity of the box
            self.halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p","rb"))
            self.halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p","rb"))
            self.ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p", "rb" ))
            self.density_field = pickle.load( open(f"{newpath}/Density_field_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p", "rb" ))
            
            #Picking up all the masss bins
            Mass_bins = np.unique(self.halo_mass)
            n_Mass_bins = len(Mass_bins)
            print("\nMass bins: ", Mass_bins)
            
            with open(f'{txt_files}/Additional_data_{I}_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.txt', 'a') as f:
                f.write(f'Mass bins {Mass_bins} \n')
                
                
            halo_data_columns = ["Base", "Order", "No. of halos", "Halo Mass"]
            halo_opted = []
            
            with open(f"{txt_files}/Halos_for_skewers_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.txt", 'w') as f:
                for i in tqdm(range(0,n_Mass_bins,int(n_Mass_bins/5))):
                    new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass = self.halo_data(i)
                    halo_opted.append([base_halo_mass, o_halo_mass, n_halos, np.log10(new_halo_mass[0])])
                    f.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
                
                if (n_Mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
                    i = n_Mass_bins-1     
                    new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass = self.halo_data(i)
                    halo_opted.append([base_halo_mass, o_halo_mass, n_halos, np.log10(new_halo_mass[0])])
                    f.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
            
            print("\n",tabulate(halo_opted, headers=halo_data_columns))     
                
            process = []
            print("\nCalculating the sightlines")
            for i in tqdm(range(0,n_Mass_bins,int(n_Mass_bins/5))):
                # create a process
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,I))
                p.start()
                process.append(p)
            
            if (n_Mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
                i = n_Mass_bins-1    
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,I))
                p.start()
                process.append(p)
                
            for p in process:
                p.join()
                
                
