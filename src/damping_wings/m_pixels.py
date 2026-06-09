#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:58:07 2022

@author: sharma

This code calculates the average mass of the pixel of the box to ensure the minimum halo mass is greater than pixel mass
"""

#-----------------------------------------------------------------------------
import py21cmfast as p21c
import numpy as np
from .config.constants import H0, Omega_m, Omega_lambda, Omega_k, G, Conversion_kg_Solar_mass, Conversion_m_to_Mpc, L_Box, DIM  
from .config.parameters_file import Parameters    
from .utils import H

#-----------------------------------------------------------------------------
def Get_me_M_min(initial_conditions: p21c.InitialConditions) -> float:
    '''
    This function calculates the average mass of the pixel of the box for a given set of box parameters

    Parameters
    -------
    initial_conditions : p21c.InitialConditions
        Initial conditions from the simulation box

    Returns
    -------
    M_pixels : float
        Returns the average mass of the pixel of the box for a given set of box parameters

    '''

    #-----------------------------------------------------------------------------
    #Setting up the ionized box at the given redshift and initial conditions

    avg_den: float  # Normalized average density of the box
    rho_c: float  # Critical density in Msun/Mpc^3
    rho_avg: float # Average baryonic density in Msun/Mpc^3
    M_pixels: float  # Pixel mass in Msun

    perturbed_field = p21c.perturb_field(
        redshift = Parameters['z'],
        initial_conditions = initial_conditions
        )
    
    avg_den = perturbed_field.get("density").mean()
    
    # Critical density
    rho_c = ((3*H(Parameters['z'])**2)/(8*np.pi*G))*(Conversion_kg_Solar_mass/Conversion_m_to_Mpc)
    rho_avg = Omega_m*rho_c*(avg_den+1) 

    M_pixels = rho_avg*(L_Box/(DIM*(1+Parameters['z'])))**3
    return float(M_pixels)


if __name__ == '__main__':
    pass
    

