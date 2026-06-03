"""
Created on Mon Mar 20 15:12:17 2023

@author: sharma
"""
import numpy as np
from numpy.typing import NDArray
import sys
from multiprocessing import Process
from tabulate import tabulate
import time
import matplotlib.pyplot as plt
import py21cmfast as p21c

#-----------------------------------------------------------------------------
# Models
from damping_wings.config.constants import L_Box, HII_DIM, DIM, seed
from damping_wings.config import parameters_file as params
from damping_wings.m_pixels import Get_me_M_min
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    cache_obj = p21c.OutputCache(cache_path)

    #-----------------------------------------------------------------------------
    # Setting up initial conditions
    print('\nSetting up initial conditions of the box')

    new_inputs = p21c.InputParameters(
        simulation_options = {"DIM": DIM , "HII_DIM": HII_DIM, "BOX_LEN": L_Box},
        matter_options = {"USE_FFTW_WISDOM": True, "PERTURB_ALGORITHM": "2LPT", "SOURCE_MODEL": "E-INTEGRAL"},
        # matter_options = {"PERTURB_ALGORITHM": "2LPT"},
        astro_options= {"M_MIN_in_Mass": True, "USE_EXP_FILTER": False, "USE_UPPER_STELLAR_TURNOVER": False},
        cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, OMm = 0.3, OMb = 0.045),
        random_seed=seed
        )

    initial_conditions = p21c.compute_initial_conditions(
        inputs=new_inputs, cache = cache_obj, write=True
    )

    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    start_time = time.perf_counter()
    m_p = mp.Get_me_M_min(initial_conditions)        # Average pixel mass; run once
    time_elapsed = time.perf_counter() - start_time

    #-----------------------------------------------------------------------------

    m_min = float('{:.2f}'.format(np.log10(m_p*20)))
    print("m_min = ", m_min)

    #-----------------------------------------------------------------------------
    nn = 2   # Number of points for each parameter
    param_ranges ={
        'z_list': np.array([6,7]),    # Redshift
        'm_min_list': np.linspace(m_min-1,m_min+1,nn),    # Minimum mass of star forming halos
        'target_xh_list': np.linspace(0.25,0.75,nn),     # Mean neutral fraction of the box
        'alpha_esc_list': np.linspace(-2,0,nn),    # alpha escape
        'alpha_star_list': np.linspace(0,1,nn),    # alpha star
        'f_star_list': np.linspace(-2,-0.25,nn),   # f star
        'tq_list': np.linspace(0,params.Parameters['tq']*30,nn)   # Quasar lifetime
        }

    model = Models(param_ranges)
    model.modelling(initial_conditions, cache_obj)

    r = []        
    r.append(model.rank_calculation(P_VO=["M_min","target_xh"], Avoid_mid = True))

    process = []
    print("\nRunning Models")
    for i in r:
        p = Process(target= model.modelling, args=(initial_conditions,cache_obj[i]))
        p.start()
        process.append(p)
        
    for p in process:
        p.join()

    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    ## Section to calibrate with respect to one variable to find the effective value of the
    # mass_bins = np.unique(model.halo_mass)
    # n_mass_bins = len(mass_bins)

    # calibration_variable = 'target_xh'

    # target_variable = {'tq':params.Parameters['tq']}
    # target_variable = {'tq':0.0}
    
    # effective_xh = model.calibrating_variables(calibration_variable, target_variable, initial_conditions)
    
