#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:20:11 2022

@author: sharma

Description: This code is to set up the parameters, run all the other codes (parallely or non-parallely), and plot the final results
"""
import numpy as np
import sys
# To monitor the processing speed
import time
#from mpi4py import MPI
import logging
import py21cmfast as p21c
# For interacting with the cache
from py21cmfast import cache_tools
#-----------------------------------------------------------------------------
# Models
# Constants
from constants import *
# Calculating the average mass of the pixel of the box and making sure that Minimum halo mass is at least 20 times more than the pixel mass
import calculating_m_pixels as MP
# Parameters file
import parameters_file as Params
from optimized_code_running_models import Models
import plotting_wings as PW

#-----------------------------------------------------------------------------
# Logger module
logger = logging.getLogger("Damping_wings")

logFileFormatter = logging.Formatter(
    fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
fileHandler = logging.FileHandler(filename=f'{newpath}/logging.log')
fileHandler.setFormatter(logFileFormatter)
fileHandler.setLevel(level=logging.INFO)

logger.addHandler(fileHandler)

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# cache_file = r'/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/21cmfast_cache' 
# if not os.path.exists(cache_file):
#     os.makedir(cache_file)
    
# p21c.config['direc'] = cache_file
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
 #Setting up initial conditions
print('\nSetting up initial conditions of the box')

initial_conditions = p21c.initial_conditions(
    user_params = {"DIM": DIM , "HII_DIM": HII_DIM, "BOX_LEN": L_Box, "N_THREADS": 32, "USE_FFTW_WISDOM": True, "USE_2LPT":False},
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, OMm = 0.3, OMb = 0.045),
    random_seed=54321
    )

#-----------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Clearing not required cache
cache_tools.clear_cache(direc = cache_file)

#----------------------------------------------------------------------------------------------------------------------------------------------------------


start_time = time.perf_counter()
M_p = MP.Get_me_M_min(initial_conditions)        # Average pixel mass
time_elapsed = time.perf_counter() - start_time
logger.info(f"Pixel mass = {M_p}")
logger.info(f"It took {time_elapsed} seconds to calculate the pixel mass")

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

M_min = float('{:.2f}'.format(np.log10(M_p*20)))

#-----------------------------------------------------------------------------

# Range over which we will vary our parameter space
NN = 2   # Number of points for each parameter
Param_Ranges ={
    
    'z_list': np.linspace(6,8,NN),    # Redshift
    'M_min_list': np.linspace(M_min,M_min+1,NN),    # Minimum mass of star forming halos
    'target_xh_list': np.linspace(0.1,0.9,NN),     # Mean neutral fraction of the box
    'alpha_esc_list': np.linspace(-1,0,NN),    # alpha escape
    'alpha_star_list': np.linspace(0,1,NN),    # alpha star
    'f_star_list': np.linspace(-2,-0.25,NN),   # f star
    'tq_list': np.linspace(0,Params.Parameters['tq']*10**2,NN)   # Quasar lifetime
    }
# rank of a specific parameter set is rank = ∑I_j*n^(D-j) for j in (1,D)
Param_Ranges
m = Models(Param_Ranges)
# m.rank_list
        
r1 = m.rank_calculation()
m.modelling(initial_conditions)

