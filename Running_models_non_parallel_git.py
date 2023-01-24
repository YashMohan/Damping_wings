#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:20:11 2022

@author: sharma

Description: This code is to set up the parameters, run all the other codes (parallely or non-parallely), and plot the final results
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
import os
import sys
import importlib
from matplotlib import gridspec
# To monitor the processing speed
import time
#from mpi4py import MPI

#-----------------------------------------------------------------------------
# Models
# Constants
from Constants import *
# Calculating the average mass of the pixel of the box and making sure that Minimum halo mass is at least 20 times more than the pixel mass
import Calculating_M_pixels as MP
# Parameters file
import Parameters_file as Params
import Corner_models as CM
import Face_models as FM
import Middle_model as MM
#-----------------------------------------------------------------------------

logging_file = open(newpath+"/Time_logging.txt", "w")

#sys.stdout = open(newpath+'/Output_logging.txt', 'w')

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

# Note: Parameters that we are playing with are : redshift 'z', global mean neutral fraction 'xh' (calibrated by f_esc), alpha star, alpha escape, f star, Turn over mass 'M_min', and the quasar lifetime 'tq'

#Rough Sketch
'''
rough sketch : the bounds on the parameters are taken from Greig 2019:
    
    z = 6 to 8
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
start_time = time.time()
M_p = MP.Get_me_M_min()        # Average pixel mass
time_elapsed = time.time() - start_time
#print(time_elapsed)
logging_file.write(f'It took {time_elapsed} seconds to calculate the pixel mass')

M_min = float('{:.2f}'.format(np.log10(M_p*20)))

#-----------------------------------------------------------------------------
Param_Ranges ={
    
    'z_list': np.linspace(6,8,2),    # Redshift
    'M_min_list': np.linspace(M_min,M_min+1,2),    # Minimum mass of star forming halos
    'target_xh_list': np.linspace(0.1,0.9,2),     # Mean neutral fraction of the box
    'alpha_esc_list': np.linspace(-1,0,2),    # alpha escape
    'alpha_star_list': np.linspace(0,1,2),    # alpha star
    'f_star_list': np.linspace(-2,-0.25,2),   # f star
    'tq_list': np.linspace(0,3.154*10**13,2)     # Quasar lifetime
    }

Default_D = len(Params.Parameters) - 1
D = len(Param_Ranges)  # Dimension of corner models, i.e., total corner models = 2^D, corners of D-dim hypercube

if D<Default_D:     # Just in case our list of Parameters we want to vary is less than the total number of parameters we have, then this condition will make sure our corner and face model's ranks don't overlap
    D = Default_D
    
#-----------------------------------------------------------------------------
# Running all corner models
print("Parameters and their ranges\n", Param_Ranges)
corner_models_processing_time = CM.get_corner_models(Param_Ranges)

# logging_file.write(f"It took {corner_models_processing_time} to calculate all the corner models")

#-----------------------------------------------------------------------------
# Running all face models

variables_list = ['M_min', 'target_xh', 'alpha_esc', 'alpha_star', 'f_star']  # List of variables we need for face models
d = len(variables_list)

variables_list2 = [f"{k}_list" for k in variables_list]         # adding _list suffix to compare with Param_Ranges
variables_ranges = [Param_Ranges[k] for k in variables_list2 if k in Param_Ranges]    # Selecting the parameter ranges of the desired list of variables

face_models_processing_time = FM.get_face_models(variables_list, variables_ranges, D)

# logging_file.write(f"It took {face_models_processing_time} to calculate all the face models")
                     
#-----------------------------------------------------------------------------
# Running middle model
# Middle model is just the model with default parameters, it has rank = -1 which is the default value

middle_model_processing_time = MM.get_face_models()

# logging_file.write(f"It took {middle_model_processing_time} to calculate the middle models")
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
FM.Comprehensive_plots(d, D)
        


#sys.stdout.close()
# Other approaches to calculate variables_list2
#variables_list2 = [z_list,target_xh_list,alpha_esc_list,alpha_star_list,f_star_list]  # Manual procedure
#variables_list2 = [Param_Ranges[k] for k in Param_Ranges.keys() & set(variables_list2)] # set re-orders variables_list2 alphabetically 

