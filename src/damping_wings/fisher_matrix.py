import numpy as np
from numpy.typing import NDArray
from typing import Optional
import sys
from ordered_set import OrderedSet
import itertools
import pickle
from multiprocessing import Process
from tabulate import tabulate
import time
import h5py
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import chisquare
from scipy.stats import qmc
from scipy import stats
import random
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ipywidgets import interact
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import copy
from matplotlib import rcParams
import numba
from numba import njit, prange
import corner
from IPython import embed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from normal_corner import normal_corner


from .config.constants import SimParams, SimParamsRanges, N_sightlines, newpath, H0, Omega_b, Omega_k, Omega_lambda, Omega_m, Nion, Conversion_amu_Mpc, L_Box, HII_DIM, DIM, txt_files, c, seed
from .config.parameters_file import Parameters as params 
from .modelling import Models 

def add_proxy(
    clean_data: NDArray[np.float64],
    z: float, 
    R: NDArray[np.float64], 
    seed: int) -> NDArray[np.float64]:
    """This function adds the Lyman alpha resonance absorption within the proximity zone of a quasar

    Args:
        clean_data (NDArray[np.float64]): Raw damping wing profiles from the simulations
        z (float): redshift of the quasar
        R (NDArray[np.float64]): The region of Proximity zone
        seed (int): simulation seed

    Returns:
        NDArray[np.float64]: Modified raw data with the effects of Lyman alpha resonance absorption within the proximity zone of a quasar
    """
    np.random.seed(seed)
    
    GammaHI: NDArray[np.float64]    # photoionization rate
    tauLya: NDArray[np.float64]     # Lyman alpha optical depth
    siglntau: NDArray[np.float64]   # This adds the pixel variations on the added proximity zones
    
    if params.Parameters['z'] == 7.0:
        print("Redshift, Nion: ", params.Parameters['z'], np.log10(Nion), (Nion/10**57))
        GammaHI = 1.85e-11*(Nion/10**57)*(R/(1+z))**-2 # z= 7
        tauLya = 9.27*(GammaHI/2.5e-13)**-0.55 # z = 7

    elif params.Parameters['z'] == 6.5:
        print("Redshift: ", params.Parameters['z'])
        GammaHI = 1.85e-11*(R/(1+z))**-2 # z = 6.5
        tauLya = 7.5*(GammaHI/2.5e-13)**-0.55  # z = 6.5
   
    elif params.Parameters['z'] == 6.0:
        print("Redshift: ", params.Parameters['z'])
        GammaHI = 3.2e-11*(R/(1+z))**-2 # z = 6
        tauLya = 5.7*(GammaHI/2.5e-13)**-0.55  # z= 6 
    
    siglntau = 0.7 # 1.0
    
    for i in range(len(clean_data)):
        clean_data[i,0:len(R)] = clean_data[i,0:len(R)] + np.exp(np.random.normal(np.log(tauLya),siglntau))
    
    return clean_data

def add_noise(
    clean_data: NDArray[np.float64], 
    SNR_A: float, 
    SNR_M: float, 
    seed: int) -> NDArray[np.float64]:
    """This function modifies the given clean data with some gaussian/ normal additive and multiplicative noise of
    a desired signal to noise ratio

    Args:
        clean_data (NDArray[np.float64]): Raw damping wing profiles from the simulations, updated with the Lyman alpha resonance absorption within the proximity zone of a quasar
        SNR_A (float): Signal to Noise ratio for the additive/ spectral noise
        SNR_M (float): Signal to Noise ratio for the multiplicative/ continuum noise
        seed (int): simulation seed

    Returns:
        NDArray[np.float64]: Noisy data, updated with spectral and continuum noise
    """
    np.random.seed(seed)
    
    signal_power: int = 1
    noise_power_A: float = signal_power/SNR_A
    noise_power_M: float = signal_power/SNR_M
    additive_noise: NDArray[np.float64]
    multiplicative_noise: NDArray[np.float64]
    
    additive_noise = np.random.normal(0,noise_power_A, size=clean_data.shape)
    
    multiplicative_noise = np.random.normal(1, noise_power_M, size=(clean_data.shape[0],1)) #, size=clean_data.shape)
    
    return clean_data*multiplicative_noise + additive_noise

@njit(parallel = True)    
def sampler(
    data:NDArray[np.float64], 
    N_data_points: int, 
    N_sample: int, 
    seed: int, 
    cov_flag: Optional[bool] = True) -> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    """This function takes the data of some large size N, then breaks it into N_sample of randomly sampled data, each with size N_data_points.
        For our case, we take the damping wing profiles and generate the samples to get the distribution for median profile and scatter width.
        Once the data has been sampled, the mean and variance of median damping wings and scatter-width are returned. 

    Args:
        data (NDArray[np.float64]): Data from which we will sample our new data set
        N_data_points (int): How many data elements should one sample has
        N_sample (int): Number of samples
        seed (int): simulation seed
        cov_flag (Optional[bool], optional): _description_. Defaults to True.

    Returns:
        tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]: mean and variance of median damping wings and scatter-width 
        from the set of the sampled data of N_sample median and scatter-width profiles. The median and scatter-width are calculated over N_data_points of 
        randomly selected Lyman alpha damping wing profiles.
    """
    random.seed(seed)

    sampling: int   # A random profile from the ensemble of Lyman alpha damping wing profiles
    sampled_data: NDArray[np.float64]   # A sample of Lyman alpha damping wing profiles with N_data_points elements
    sampled_median_data: NDArray[np.float64] = np.zeros((N_sample,len(data[0])))    # Median profile from the set of the sampled_data
    sampled_scatter_width: NDArray[np.float64] = np.zeros((N_sample, len(data[0]))) # Scatter-width profile from the set of the sampled_data
    up_quant:NDArray[np.float64]    # Upper profile of the 1-sigma quantile from the set of the sampled_data
    low_quant: NDArray[np.float64]  # Lower profile of the 1-sigma quantile from the set of the sampled_data
    
    for j in range(N_sample):   
        sampled_data = np.zeros((N_data_points, len(data[0])))
        
        for i in range(N_data_points):
            sampling = random.randrange(len(data[:]))
            sampled_data[i] = data[sampling]
            
        sampled_median_data[j] =  np.array([np.median(sampled_data[:,i]) for i in range(sampled_data.shape[1])]) 
        up_quant = np.array([np.quantile(sampled_data[:,i],0.84) for i in range(sampled_data.shape[1])]) 
        low_quant = np.array([np.quantile(sampled_data[:,i],0.16) for i in range(sampled_data.shape[1])]) 
        sampled_scatter_width[j] =  up_quant - low_quant

    
    mean_sampled_median_data = np.array([np.mean(sampled_median_data[:,i]) for i in range(sampled_median_data.shape[1])]) 
    var_sampled_median_data = np.array([np.var(sampled_median_data[:,i]) for i in range(sampled_median_data.shape[1])]) 
    mean_sampled_scatter_width = np.array([np.mean(sampled_scatter_width[:,i]) for i in range(sampled_scatter_width.shape[1])]) 
    var_sampled_scatter_width = np.array([np.var(sampled_scatter_width[:,i]) for i in range(sampled_scatter_width.shape[1])]) 
    
    return mean_sampled_median_data, var_sampled_median_data, mean_sampled_scatter_width, var_sampled_scatter_width;

@njit(parallel = True)
def sampler_dist(
    data:NDArray[np.float64], 
    N_data_points: int, 
    N_sample: int, 
    seed: int, 
    cov_flag: Optional[bool] = True) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """This function takes the data of some large size N, then breaks it into N_sample of randomly sampled data, each with size N_data_points.
       For our case, we take the damping wing profiles and generate the samples to get the distribution for median profile and scatter width.
       This function returns the final sampled distribution instead of just mean and variance.

    Args:
        data (NDArray[np.float64]): Data from which we will sample our new data set
        N_data_points (int): How many data elements should one sample has
        N_sample (int): Number of samples
        seed (int): simulation seed
        cov_flag (Optional[bool], optional): _description_. Defaults to True.

    Returns:
        tuple[NDArray[np.float64],NDArray[np.float64]]:  The median and scatter-width distributions of N_sample data profiles, calculated over N_data_points of 
        randomly selected Lyman alpha damping wing profiles.
    """
    random.seed(seed)
    
    sampling: int   # A random profile from the ensemble of Lyman alpha damping wing profiles
    sampled_data: NDArray[np.float64]   # A sample of Lyman alpha damping wing profiles with N_data_points elements
    sampled_median_data: NDArray[np.float64] = np.zeros((N_sample,len(data[0])))    # Median profile from the set of the sampled_data
    sampled_scatter_width: NDArray[np.float64] = np.zeros((N_sample, len(data[0]))) # Scatter-width profile from the set of the sampled_data
    up_quant:NDArray[np.float64]    # Upper profile of the 1-sigma quantile from the set of the sampled_data
    low_quant: NDArray[np.float64]  # Lower profile of the 1-sigma quantile from the set of the sampled_data
    
    for j in range(N_sample):
        
        sampled_data = np.zeros((N_data_points, len(data[0])))
        
        for i in range(N_data_points):
            sampling = random.randrange(len(data[:]))
            sampled_data[i] = data[sampling]

        sampled_median_data[j] =  np.array([np.median(sampled_data[:,i]) for i in range(sampled_data.shape[1])])
        up_quant = np.array([np.quantile(sampled_data[:,i],0.84) for i in range(sampled_data.shape[1])])
        low_quant = np.array([np.quantile(sampled_data[:,i],0.16) for i in range(sampled_data.shape[1])])
        sampled_scatter_width[j] =  up_quant - low_quant
    
    return sampled_median_data, sampled_scatter_width;

def differentiation(ranks: list[int], 
                    N_data_points: int, 
                    N_sample: int, 
                    SNR_A: float, 
                    SNR_M: float, 
                    z: NDArray[np.float64], 
                    R: NDArray[np.float64], 
                    model: object,
                    M_qso_base: int,
                    M_qso_order: int, 
                    seed: int, 
                    noise: Optional[True]=True) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """This function calculates the first order partial derivative of the data with respect to the parameters defined
       using the ranks. It calculates the derivate using central difference method.

    Args:
        ranks (list[int]): rank of the models
        N_data_points (int): How many data elements should one sample has
        N_sample (int): Number of samples
        SNR_A (float): Signal to Noise ratio for the additive/ spectral noise
        SNR_M (float): Signal to Noise ratio for the multiplicative/ continuum noise
        z (float): redshift of the quasar
        R (NDArray[np.float64]): The region of Proximity zone
        model (object): _description_
        M_qso_base (int): Halo Mass parameters for the derivative model
        M_qso_order (int): Halo Mass parameters for the derivative model
        seed (int): simulation seed
        noise (Optional[True], optional): If the noise should be added to the raw data. Defaults to True.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: _description_
    """
    tau_1
    tau_2
    e_tau_1
    e_tau_2
    noisy_e_tau_1
    noisy_e_tau_2
    mean_sampled_e_tau_1
    mean_sampled_e_tau_2
    mean_sampled_delta_SW_1
    mean_sampled_delta_SW_2
    delta_h
    parameter_name
    delta_x
    
    
    with h5py.File(f"{newpath}/skewers_HM_{M_qso_base}_{M_qso_order}_rank_{ranks[1]}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.h5", 'r') as f:
        tau_1 = f.get("tau")[:]
    
    with h5py.File(f"{newpath}/skewers_HM_{M_qso_base}_{M_qso_order}_rank_{ranks[0]}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.h5", 'r') as f:
        tau_2 = f.get("tau")[:]
        
    # Adding proximity zone
    proximity_zone_tau_1 = add_proxy(tau_1, z,R, seed)
    e_tau_1 = np.exp(-proximity_zone_tau_1)

    proximity_zone_tau_2 = add_proxy(tau_2, z,R, seed)
    e_tau_2 = np.exp(-proximity_zone_tau_2)

    if noise:
        print("adding noise to the data")
        noisy_e_tau_1 = add_noise(e_tau_1, SNR_A, SNR_M, seed)
        noisy_e_tau_2 = add_noise(e_tau_2, SNR_A, SNR_M, seed)
    else:
        print("no noise added to the data")
        noisy_e_tau_1 = e_tau_1
        noisy_e_tau_2 = e_tau_2
        
    mean_sampled_e_tau_1, _, mean_sampled_delta_SW_1, _ = sampler(noisy_e_tau_1, N_data_points, N_sample, seed)
    mean_sampled_e_tau_2, _, mean_sampled_delta_SW_2, _ = sampler(noisy_e_tau_2, N_data_points, N_sample, seed)
    
    delta_h = [ [index, elements[0]-elements[-1]] for index,elements in enumerate(zip(model.rank_list[ranks[0]],model.rank_list[ranks[1]])) if np.abs(elements[0]-elements[-1]) > 0]
    print('delta_h: ', delta_h)
    
    for differential in delta_h:
        
        parameter_name = model.Parameters_names[differential[0]]
        print("Differentiating the following parameter: ",parameter_name)
        delta_x = differential[-1]
    
        dDW_dpara = (mean_sampled_e_tau_2 - mean_sampled_e_tau_1)/delta_x
        
        dSW_dpara = (mean_sampled_delta_SW_2 - mean_sampled_delta_SW_1)/delta_x
        
        
    return  dDW_dpara, dSW_dpara

def corr_fisher_matrix(data_set: NDArray[np.float64], 
                       ddata_dtheta: NDArray[np.float64], 
                       len_theta: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ Calculating Fisher Matrix with correlated pixels

    Args:
        data_set (NDArray[np.float64]): _description_
        ddata_dtheta (NDArray[np.float64]): _description_
        len_theta (int): _description_

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Returns Fisher Matrix with correlated pixels and the corresponding covariance matrix for the parameters
    """
    
    fisher_matrix: NDArray[np.float64] = np.zeros((len_theta,len_theta))    # Fisher Matrix
    cov_matrix: NDArray[np.float64] = np.cov(data_set.transpose())          # Calculating the covariance matrix
    cov_matrix_inv: NDArray[np.float64] = np.linalg.inv(cov_matrix)         # Inverse of covariance matrix, since, FM = dtheta/dp * C^{-1} * dtheta/dp.T
    cov_fisher_matrix: NDArray[np.float64]                                  # Covariance matrix of the uncertainties is the inverse of FM
    
    fisher_matrix  = np.matmul(ddata_dtheta, np.matmul(cov_matrix_inv, ddata_dtheta.transpose()))
    cov_fisher_matrix = np.linalg.inv(fisher_matrix)
    
    return fisher_matrix, cov_fisher_matrix
    
def uncorr_fisher_matrix(variance: NDArray[np.float64], 
                         ddata_dtheta: NDArray[np.float64], 
                         len_theta: int) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Calculating Fisher Matrix with uncorrelated pixels

    Args:
        variance (NDArray[np.float64]): _description_
        ddata_dtheta (NDArray[np.float64]): _description_
        len_theta (int): _description_

    Returns:
        tuple[NDArray[np.float64],NDArray[np.float64]]: Returns Fisher Matrix with uncorrelated pixels and the corresponding covariance matrix for the parameters
    """
    fisher_matrix: NDArray[np.float64]   # Fisher Matrix
    cov_fisher_matrix: NDArray[np.float64]   # Covariance matrix for the uncertainties is the inverse of FM
    element_UFM: NDArray[np.float64]    # FM = element_UFM x element_UFM (cartesian product of each elements of the element_UFM matrix = dP/dt_i*dP/dt_j)
    
    element_UFM = ddata_dtheta/np.sqrt(variance)
    element_UFM = [element_UFM,element_UFM]
    fisher_matrix = np.sum( np.prod( list( itertools.product(*element_UFM)), axis=1), axis = 1)
    fisher_matrix = np.reshape(fisher_matrix,(len_theta,len_theta)) # Reshaping it to the desired shape
    cov_fisher_matrix = np.linalg.inv(fisher_matrix)
    
    return fisher_matrix, cov_fisher_matrix
    
        
def prox_fisher_matrix(N_sample: int, 
                       N_data_points: int, 
                       SNR_A: float, 
                       SNR_M: float,
                       z: NDArray[np.float64], 
                       R: NDArray[np.float64],
                       fisher_parameters: dict[str:list,str:list,str:list,str:list], 
                       M_qso_masses: list[float,float],
                       M_qso_fiducial: list[int,int],
                       seed: int, 
                       noise: Optional[bool] =True) -> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    """_summary_

    Args:
        N_sample (int): _description_
        N_data_points (int): _description_
        SNR_A (float): _description_
        SNR_M (float): _description_
        z (NDArray[np.float64]): _description_
        R (NDArray[np.float64]): _description_
        fisher_parameters (_type_): _description_
        M_qso_masses (list[float,float]): _description_
        M_qso_fiducial (list[int,int]): _description_
        seed (int): _description_
        noise (Optional[bool], optional): _description_. Defaults to True.

    Returns:
        tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]: _description_
    """
    # Loading the fiducial data, adding the proximity zone and noise to it
    with h5py.File(f"{newpath}/skewers_HM_{M_qso_fiducial[0]}_{M_qso_fiducial[1]}_rank_0_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.h5", 'r') as f:
        lamda = f.get("lambda")[:]
        tau_fiducial = f.get("tau")[:]

    # Adding proximity zone
    proximity_zone_tau_fiducial = add_proxy(tau_fiducial, z,R, seed)
    e_tau_fiducial = np.exp(-proximity_zone_tau_fiducial)

    # Adding noise
    if noise:
        print("adding noise to the data")
        noisy_e_tau_fiducial = add_noise(e_tau_fiducial, SNR_A, SNR_M, seed)
    else:
        print("no noise added")
        noisy_e_tau_fiducial = e_tau_fiducial
   
    _, var_sampled_e_tau_fiducial, _, var_sampled_delta_SW_fiducial = sampler(noisy_e_tau_fiducial, N_data_points, N_sample, seed, cov_flag = False);

    # The sampled distribution of DW and SW
    sampled_median_data_trimmed, sampled_scatter_width_trimmed = sampler_dist(noisy_e_tau_fiducial, N_data_points, N_sample, seed);

    # The derivative variables
    dDW_dP = np.zeros((len(fisher_parameters.keys()),len(z)))
    dSW_dP = np.zeros((len(fisher_parameters.keys()),len(z)))
    iteration = 0

    for parmeters, ranks in fisher_parameters.items(): # This loop runs over all the ranks and calculates the derivative of each ranked parameter w.r.t the fiducial values 
        dDW_dP[iteration], dSW_dP[iteration] = differentiation(ranks, N_data_points, N_sample, SNR_A, SNR_M,z, R, model, seed, noise)
        iteration +=1

    # To calculate the derivative with respect to M_qso
    halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{0}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p","rb"))
    mass_bins = np.unique(halo_mass)

    o_halo_mass = [int(np.floor(np.log10(M_qso))) for M_qso in M_qso_masses ]           # The order of the mass of the halo in the form of Mass = base*10^order, this is an approximation not the exact value of the masses
    base_halo_mass = [int(np.round(10*M_qso/(10**o),0)) for M_qso, o in zip(M_qso_masses,o_halo_mass) ]
    print(base_halo_mass, o_halo_mass)

    with h5py.File(f"{newpath}/skewers_HM_{base_halo_mass[0]}_{o_halo_mass[0]}_rank_{0}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.h5", 'r') as f:
        tau_1 = f.get("tau")[:]

    with h5py.File(f"{newpath}/skewers_HM_{base_halo_mass[1]}_{o_halo_mass[1]}_rank_{0}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.h5", 'r') as f:
        tau_2 = f.get("tau")[:]

    # Adding proximity zone
    proximity_zone_tau_1 = add_proxy(tau_1, z,R, seed)
    e_tau_1 = np.exp(-proximity_zone_tau_1)

    proximity_zone_tau_2 = add_proxy(tau_2, z,R, seed)
    e_tau_2 = np.exp(-proximity_zone_tau_2)

    if noise:
        print("adding noise to the data")
        noisy_e_tau_1 = add_noise(e_tau_1, SNR_A, SNR_M, seed)
        noisy_e_tau_2 = add_noise(e_tau_2, SNR_A, SNR_M, seed)
    else:
        print("no noise is added")
        noisy_e_tau_1 = e_tau_1
        noisy_e_tau_2 = e_tau_2

    mean_sampled_e_tau_1_trimmed, _, mean_sampled_delta_SW_1_trimmed, _ = sampler(noisy_e_tau_1, N_data_points, N_sample, seed)
    mean_sampled_e_tau_2_trimmed, _, mean_sampled_delta_SW_2_trimmed, _ = sampler(noisy_e_tau_2, N_data_points, N_sample, seed)

    delta_h =  np.log(M_qso_masses[1]) - np.log(M_qso_masses[0])

    dDW_dP[-1] = (mean_sampled_e_tau_2_trimmed - mean_sampled_e_tau_1_trimmed)/delta_h
    dSW_dP[-1]= (mean_sampled_delta_SW_2_trimmed - mean_sampled_delta_SW_1_trimmed)/delta_h

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Getting the Fisher matrices
    # Case 1.1: uncorrelated pixel error Fisher Matrix:
    uncorr_fisher_matrix_DW, uncorr_cov_fisher_matrix_DW = uncorr_fisher_matrix(var_sampled_e_tau_fiducial, dDW_dP, len(fisher_parameters.keys()))
    uncorr_fisher_matrix_SW, uncorr_cov_fisher_matrix_SW = uncorr_fisher_matrix(var_sampled_delta_SW_fiducial, dSW_dP, len(fisher_parameters.keys()))

    # Case 1.2: correlated pixel error Fisher Matrix:
    corr_fisher_matrix_DW, corr_cov_fisher_matrix_DW = corr_fisher_matrix(sampled_median_data_trimmed, dDW_dP, len(fisher_parameters.keys()))
    corr_fisher_matrix_SW, corr_cov_fisher_matrix_SW = corr_fisher_matrix(sampled_scatter_width_trimmed, dSW_dP, len(fisher_parameters.keys()))

    # Case 1.3: combined fisher matrix for DW and SW with correlated pixel errors:
    combined_sampled_data = np.concatenate((sampled_median_data_trimmed,sampled_scatter_width_trimmed), axis=1)
    combined_ddata_dtheta = np.concatenate((dDW_dP,dSW_dP), axis=1)

    cov_combined_data = np.cov(combined_sampled_data.transpose())
    combined_corr_fisher_matrix, combined_corr_cov_fisher_matrix = corr_fisher_matrix(combined_sampled_data, combined_ddata_dtheta, len(fisher_parameters.keys()))

    # CORNER PLOTS 
    mean_values = np.array([params.Parameters['target_xh'], round(np.log10(model.rank_list[0][-1]/365.25/86400),2), params.Parameters['m_min'],np.log10(mass_bins[-8])])        # Change the mass_bin number here as well when changing z
    variables = [r"$\mathrm{x_{HI}}$", r"$\log\mathrm{t_{q}}$ yr", r"$\mathrm{\log M_{min}}$", r"$\mathrm{\log M_{qso}}$"]

    # combined fisher matrix for DW and SW with correlated pixel errors:
    combined_sample = np.random.multivariate_normal(mean_values, combined_corr_cov_fisher_matrix, size=10000)

    # figure_1 = corner.corner(combined_sample, labels=variables, plot_density=False, show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12})


    # DW:
    # figure_2 = normal_corner.normal_corner(corr_cov_fisher_matrix_DW, mean_values, variables)
    DW_sample = np.random.multivariate_normal(mean_values, corr_cov_fisher_matrix_DW, size=10000)

    # figure_2 = corner.corner(DW_sample, labels=variables, plot_density=False, show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12})


    # SW:
    # figure_3 = normal_corner.normal_corner(corr_cov_fisher_matrix_SW, mean_values, variables)
    SW_sample = np.random.multivariate_normal(mean_values, corr_cov_fisher_matrix_SW, size=10000)

    # figure_3 = corner.corner(SW_sample, labels=variables, plot_density=False, show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12})

    sigma_levels = [0.68, 0.95]  # Corresponding to 1σ, 2σ, 3σ- 0.997
    
    fig_DW = corner.corner(
        DW_sample,
        levels=sigma_levels,
        plot_density=False,       # Show density plot
        plot_contours=True,      # Show contours
        fill_contours=True,      # Fill the contours
        plot_datapoints=False,   # No data points
        bins=30,                 # Number of bins in histograms
        labels=variables,  # Label axes
        show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12},
        smooth = 1.0,
        color = "b",
    )
    axes = np.array(fig_DW.axes).reshape(len(fisher_parameters.keys()), len(fisher_parameters.keys()))  # Adjust for the number of parameters
    axes[0,2].set_title(f"Damping Wing SNR_A {SNR_A} SNR_M {SNR_M}")
    plt.savefig(f'{plotpath}/DW_contour_plot_SNR_A_{SNR_A}SNR_M_{SNR_M}_N_{N_data_points}_N_sample_{N_sample}.png')

    fig_SW = corner.corner(
        SW_sample,
        levels=sigma_levels,
        plot_density=False,       # Show density plot
        plot_contours=True,      # Show contours
        fill_contours=True,      # Fill the contours
        plot_datapoints=False,   # No data points
        bins=30,                 # Number of bins in histograms
        labels=variables,  # Label axes
        show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12},
        smooth = 1.0,
        color = "r",
    )
    axes = np.array(fig_SW.axes).reshape(len(fisher_parameters.keys()), len(fisher_parameters.keys()))  # Adjust for the number of parameters
    axes[0,2].set_title(f"Scatter Width SNR_A {SNR_A} SNR_M {SNR_M}")
    plt.savefig(f'{plotpath}/SW_contour_plot_SNR_A_{SNR_A}SNR_M_{SNR_M}_N_{N_data_points}_N_sample_{N_sample}.png')

    fig_combined = corner.corner(
        combined_sample,
        levels=sigma_levels,
        plot_density=True,       # Show density plot
        plot_contours=True,      # Show contours
        fill_contours=True,      # Fill the contours
        plot_datapoints=False,   # No data points
        bins=30,                 # Number of bins in histograms
        labels=variables,  # Label axes
        show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12},
        smooth = 1.0,
        color = "orange",
    )
    axes = np.array(fig_combined.axes).reshape(len(fisher_parameters.keys()), len(fisher_parameters.keys()))  # Adjust for the number of parameters
    axes[0,2].set_title(f"Combined SNR_A {SNR_A} SNR_M {SNR_M}")
    plt.savefig(f'{plotpath}/Combined_contour_plot_SNR_A_{SNR_A}SNR_M_{SNR_M}_N_{N_data_points}_N_sample_{N_sample}.png')
    plt.show()
    
    fig_multiple = corner.corner(
    DW_sample,
    labels=variables,  # Label axes
    levels=sigma_levels,
    plot_density=False,       # Show density plot
    plot_contours=True,      # Show contours
    fill_contours=True,      # Fill the contours
    plot_datapoints=False,   # No data points
    bins=30,                 # Number of bins in histograms
    color = "b",
    show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12},
    smooth = 1.0,
    hist2d_kwargs = {"plot_datapoints":False, "alpha":0.5}
    #labels="DW"
    )

    # Overlay the second dataset on the same axes
    corner.corner(
        SW_sample,
        fig=fig_multiple,  # Use the existing figure
        plot_contours=True,
        fill_contours=True,
        plot_datapoints=False,   # No data points
        bins=30,
        color = "r",
        show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12},
        smooth = 1.0,
        hist2d_kwargs = {"plot_datapoints":False, "alpha":0.5}
        #labels="SW"
    )

    corner.corner(
        combined_sample,
        fig=fig_multiple,  # Use the existing figure
        plot_contours=True,
        fill_contours=True,
        plot_datapoints=False,   # No data points
        bins=30,
        color = "orange",
        show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12},
        smooth = 1.0,
        hist2d_kwargs = {"plot_datapoints":False, "alpha":0.5}
        #labels="Combination"
    )

    # Add legend
    axes = np.array(fig_multiple.axes).reshape(len(fisher_parameters.keys()), len(fisher_parameters.keys()))  # Adjust for the number of parameters
    legend_elements = [
        Patch(color="b", label="DW"),
        Patch(color="r", label="SW"),
        Patch(color="orange", label="Combination"),
    ]
    axes[1, 2].legend(
        handles=legend_elements, loc="upper right", fontsize=10,
    )
    plt.savefig(f'{plotpath}/smooth_multiple_contour_plot_combined_SNR_A_{SNR_A}SNR_M_{SNR_M}_N_{N_data_points}_N_sample_{N_sample}.png')
    plt.show()

    return corr_cov_fisher_matrix_DW, corr_cov_fisher_matrix_SW, combined_corr_fisher_matrix, combined_corr_cov_fisher_matrix
    
    