import numpy as np
from numpy.typing import NDArray
from typing import Optional
import sys
import itertools
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.figure import Figure
from numba import njit
import corner


from .config.constants import SimParams, SimParamsRanges, N_sightlines, newpath, plotpath, H0, Omega_b, Omega_k, Omega_lambda, Omega_m, Nion, Conversion_amu_Mpc, L_Box, HII_DIM, DIM, txt_files, c, seed
from .config.parameters_file import Parameters as params 
from .modelling_class import Models 

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
    
    clean_data = clean_data.copy()
    GammaHI: NDArray[np.float64]    # photoionization rate
    tauLya: NDArray[np.float64]     # Lyman alpha optical depth
    siglntau: NDArray[np.float64]   # This adds the pixel variations on the added proximity zones
    
    if params['z'] == 7.0:
        print("Redshift, Nion: ", params['z'], np.log10(Nion), (Nion/10**57))
        GammaHI = 1.85e-11*(Nion/10**57)*(R/(1+z))**-2 # z= 7
        tauLya = 9.27*(GammaHI/2.5e-13)**-0.55 # z = 7

    elif params['z'] == 6.5:
        print("Redshift: ", params['z'])
        GammaHI = 1.85e-11*(R/(1+z))**-2 # z = 6.5
        tauLya = 7.5*(GammaHI/2.5e-13)**-0.55  # z = 6.5
   
    elif params['z'] == 6.0:
        print("Redshift: ", params['z'])
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
    seed: int) -> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
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
    np.random.seed(seed)

    sampling: int   # A random profile from the ensemble of Lyman alpha damping wing profiles
    sampled_data: NDArray[np.float64]   # A sample of Lyman alpha damping wing profiles with N_data_points elements
    sampled_median_data: NDArray[np.float64] = np.zeros((N_sample,len(data[0])))    # Median profile from the set of the sampled_data
    sampled_scatter_width: NDArray[np.float64] = np.zeros((N_sample, len(data[0]))) # Scatter-width profile from the set of the sampled_data
    up_quant:NDArray[np.float64]    # Upper profile of the 1-sigma quantile from the set of the sampled_data
    low_quant: NDArray[np.float64]  # Lower profile of the 1-sigma quantile from the set of the sampled_data
    
    for j in range(N_sample):   
        sampled_data = np.zeros((N_data_points, len(data[0])))
        
        for i in range(N_data_points):
            sampling = np.random.randint(0, len(data))
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
    seed: int) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
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
    np.random.seed(seed)
    
    sampling: int   # A random profile from the ensemble of Lyman alpha damping wing profiles
    sampled_data: NDArray[np.float64]   # A sample of Lyman alpha damping wing profiles with N_data_points elements
    sampled_median_data: NDArray[np.float64] = np.zeros((N_sample,len(data[0])))    # Median profile from the set of the sampled_data
    sampled_scatter_width: NDArray[np.float64] = np.zeros((N_sample, len(data[0]))) # Scatter-width profile from the set of the sampled_data
    up_quant:NDArray[np.float64]    # Upper profile of the 1-sigma quantile from the set of the sampled_data
    low_quant: NDArray[np.float64]  # Lower profile of the 1-sigma quantile from the set of the sampled_data
    
    for j in range(N_sample):
        
        sampled_data = np.zeros((N_data_points, len(data[0])))
        
        for i in range(N_data_points):
            sampling = np.random.randint(0, len(data))
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
                    noise: Optional[bool]=True) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        model (object): Instantiated Models object containing the parameter grid and rank list
        M_qso_base (int): Halo Mass parameters for the derivative model
        M_qso_order (int): Halo Mass parameters for the derivative model
        seed (int): simulation seed
        noise (Optional[bool], optional): If the noise should be added to the raw data. Defaults to True.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Returns the derivative of the transmission and scatter-width profiles for the given parameter (calculated
        via rank)
    """
    tau_1: NDArray[np.float64]                      # Optical depth of the model perturbed to the left for derivative
    tau_2: NDArray[np.float64]                      # Optical depth of the model perturbed to the right for derivative
    e_tau_1: NDArray[np.float64]                    # Transmission profile of the model perturbed to the left for derivative
    e_tau_2: NDArray[np.float64]                    # Transmission profile of the model perturbed to the right for derivative
    noisy_e_tau_1: NDArray[np.float64]              # Noisy transmission profile of the model perturbed to the left for derivative
    noisy_e_tau_2: NDArray[np.float64]              # Noisy transmission profile of the model perturbed to the right for derivative
    mean_sampled_e_tau_1: NDArray[np.float64]       # Mean sampled transmission profile of the model perturbed to the left for derivative
    mean_sampled_e_tau_2: NDArray[np.float64]       # Mean sampled transmission profile of the model perturbed to the right for derivative
    mean_sampled_delta_SW_1: NDArray[np.float64]    # Mean sampled scatter-width profile of the model perturbed to the left for derivative
    mean_sampled_delta_SW_2: NDArray[np.float64]    # Mean sampled scatter-width profile of the model perturbed to the right for derivative
    delta_h: list[list]                             # stores the delta_x denominator for derivative 
    
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
    
    if not delta_h:
        raise ValueError(f"No parameter difference found between ranks {ranks[0]} and {ranks[1]}. "
                     f"Check that the ranks correspond to different parameter values.")
    
    for differential in delta_h:
        dDW_dpara = (mean_sampled_e_tau_2 - mean_sampled_e_tau_1)/differential[-1]
        dSW_dpara = (mean_sampled_delta_SW_2 - mean_sampled_delta_SW_1)/differential[-1]
        
        
    return  dDW_dpara, dSW_dpara

def compute_corr_fisher_matrix(data_set: NDArray[np.float64], 
                       ddata_dtheta: NDArray[np.float64], 
                       len_theta: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ Calculating Fisher Matrix with correlated pixels

    Args:
        data_set (NDArray[np.float64]): The distribution of profiles/ Observables for the Fisher matrix calculation
        ddata_dtheta (NDArray[np.float64]): The derivative of the observables with respect to the parameters
        len_theta (int): Number of observables

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
    
def compute_uncorr_fisher_matrix(variance: NDArray[np.float64], 
                         ddata_dtheta: NDArray[np.float64], 
                         len_theta: int) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Calculating Fisher Matrix with uncorrelated pixels

    Args:
        variance (NDArray[np.float64]): Variance of the sampled distribution of the observables
        ddata_dtheta (NDArray[np.float64]): The derivative of the observables with respect to the parameters
        len_theta (int): Number of observables

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

def plotting_fisher_matrix(sample: NDArray[np.float64]) -> Figure:
    """ This function generates and returns the contour plot of the given multivariate function

    Args:
        sample (NDArray[np.float64]): Sample multivariate distribution that needs to be plotted

    Returns:
        fig: contour plot
    """
    
    variables = [r"$\mathrm{x_{HI}}$", r"$\log\mathrm{t_{q}}$ yr", r"$\mathrm{\log M_{min}}$", r"$\mathrm{\log M_{qso}}$"]
    sigma_levels = [0.68, 0.95]  # Corresponding to 1σ, 2σ, 3σ- 0.997
    
    fig = corner.corner(
        sample,
        levels=sigma_levels,
        plot_density=False,       # Show density plot
        plot_contours=True,      # Show contours
        fill_contours=True,      # Fill the contours
        plot_datapoints=False,   # No data points
        bins=30,                 # Number of bins in histograms
        labels=variables,  # Label axes
        show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12},
        smooth = 1.0,
    )
    
    return fig
    
        
def fisher_matrix(N_sample: int, 
                  N_data_points: int, 
                  SNR_A: float, 
                  SNR_M: float,
                  z: NDArray[np.float64], 
                  R: NDArray[np.float64],
                  fisher_parameters: dict[str, list[int]], 
                  M_qso_masses: list[float],
                  M_qso_fiducial: list[int],
                  model: object,
                  seed: int, 
                  noise: Optional[bool] =True) -> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    """_summary_

    Args:
        N_sample (int): Number of samples
        N_data_points (int): How many data elements should one sample has
        SNR_A (float): Signal to Noise ratio for the additive/ spectral noise
        SNR_M (float): Signal to Noise ratio for the multiplicative/ continuum noise
        z (NDArray[np.float64]): redshift of the quasar
        R (NDArray[np.float64]): The region of Proximity zone
        fisher_parameters (dict[str, list[int]]): Parameters and their corresponding ranks for the Fisher matrix calculation
        M_qso_masses (list[float,float]): Halo Mass parameters for the derivative model
        M_qso_fiducial (list[int,int]): Halo Mass parameters for the fiducial model
        model (object): Instantiated Models object containing the parameter grid and rank list
        seed (int): simulation seed
        noise (Optional[bool], optional): _description_. optional): If the noise should be added to the raw data. Defaults to True.

    Returns:
        tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]: Returns the covariance of the parameters from transmission profiles, 
        scatter-width profiles, and a combination of both, and the Fisher matrix from the combination of transmission and scatter-width profiles
    """

    lamda: NDArray[np.float64]                          # Observed wavelength
    tau_fiducial: NDArray[np.float64]                   # Optical depth of the fiducial model
    tau_1: NDArray[np.float64]                          # Optical depth of the M_qso model perturbed to the left for derivative
    tau_2: NDArray[np.float64]                          # Optical depth of the M_qso model perturbed to the right for derivative
    proximity_zone_tau_fiducial: NDArray[np.float64]    # Optical depth of the fiducial model modified with proximity zone effects
    proximity_zone_tau_1: NDArray[np.float64]           # Optical depth of the M_qso model modified with proximity zone effects, perturbed to the left for derivative
    proximity_zone_tau_2: NDArray[np.float64]           # Optical depth of the M_qso model modified with proximity zone effects, perturbed to the right for derivative
    e_tau_fiducial: NDArray[np.float64]                 # Transmission profile of the fiducial model
    e_tau_1: NDArray[np.float64]                        # Transmission profile of the M_qso model perturbed to the left for derivative
    e_tau_2: NDArray[np.float64]                        # Transmission profile of the M_qso model perturbed to the right for derivative
    noisy_e_tau_fiducial: NDArray[np.float64]           # Noisy transmission profile of the fiducial model
    noisy_e_tau_1: NDArray[np.float64]                  # Noisy transmission profile of the M_qso model perturbed to the left for derivative
    noisy_e_tau_2: NDArray[np.float64]                  # Noisy transmission profile of the M_qso model perturbed to the right for derivative
    var_sampled_e_tau_fiducial: NDArray[np.float64]     # Variance of the sampled median transmission profile distribution
    var_sampled_delta_SW_fiducial: NDArray[np.float64]  # Variance of the sampled scatter-width profile distribution
    sampled_median_data: NDArray[np.float64]            # Distribution of the median transmission profiles
    sampled_scatter_width: NDArray[np.float64]          # Distribution of the scatter-width transmission profiles
    dDW_dP: NDArray[np.float64]                         # Derivatives of transmission profiles with respect to the individual parameters
    dSW_dP: NDArray[np.float64]                         # Derivatives of scatter-width profiles with respect to the individual parameters
    iteration: int                                      # iteration over parameters
    o_halo_mass: list[int]                              # Halo mass parameters for the M_qso derivative
    base_halo_mass: list[int]                           # Halo mass parameters for the M_qso derivative
    mean_sampled_e_tau_1: NDArray[np.float64]           # Mean sampled transmission profile of the M_qso model perturbed to the left for derivative
    mean_sampled_e_tau_2: NDArray[np.float64]           # Mean sampled transmission profile of the M_qso model perturbed to the right for derivative
    mean_sampled_delta_SW_1: NDArray[np.float64]        # Mean sampled scatter-width profile of the M_qso model perturbed to the left for derivative
    mean_sampled_delta_SW_2: NDArray[np.float64]        # Mean sampled scatter-width profile of the M_qso model perturbed to the right for derivative
    delta_h: float                                      # delta_x denominator for M_qso derivative
    corr_fisher_matrix_DW: NDArray[np.float64]          # Pixel correlated Fisher matrix for the transmission profiles
    corr_cov_fisher_matrix_DW: NDArray[np.float64]      # Pixel correlated covariance matrix of the parameters of transmission profiles
    corr_fisher_matrix_SW: NDArray[np.float64]          # Pixel correlated Fisher matrix for the scatter-width profiles
    corr_cov_fisher_matrix_SW: NDArray[np.float64]      # Pixel correlated covariance matrix of the parameters of transmission and scatter-width profiles
    uncorr_fisher_matrix_DW: NDArray[np.float64]          # Uncorrelated Fisher matrix for the transmission profiles
    uncorr_cov_fisher_matrix_DW: NDArray[np.float64]    # Uncorrelated covariance matrix of the parameters of transmission profiles
    uncorr_fisher_matrix_SW: NDArray[np.float64]        # Uncorrelated Fisher matrix for the scatter-width profiles
    uncorr_cov_fisher_matrix_SW: NDArray[np.float64]    # Uncorrelated covariance matrix of the parameters of transmission and scatter-width profiles
    combined_sampled_data: NDArray[np.float64]          # Combination of the transmission and scatter-width profiles
    combined_ddata_dtheta: NDArray[np.float64]          # Derivative of the transmission and scatter-width profiles combination
    combined_corr_fisher_matrix: NDArray[np.float64]    # Pixel correlated Fisher matrix for the combination of transmission and scatter-width profiles
    combined_corr_cov_fisher_matrix: NDArray[np.float64]# Pixel correlated covariance matrix of the parameters of transmission and scatter-width profiles
    mean_values: NDArray[np.float64]                    # The true values of the parameters
    
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
   
    _, var_sampled_e_tau_fiducial, _, var_sampled_delta_SW_fiducial = sampler(noisy_e_tau_fiducial, N_data_points, N_sample, seed);

    # The sampled distribution of DW and SW
    sampled_median_data, sampled_scatter_width = sampler_dist(noisy_e_tau_fiducial, N_data_points, N_sample, seed);

    # The derivative variables
    dDW_dP = np.zeros((len(fisher_parameters.keys()),len(z)))
    dSW_dP = np.zeros((len(fisher_parameters.keys()),len(z)))
    iteration = 0

    for parmeters, ranks in fisher_parameters.items(): # This loop runs over all the ranks and calculates the derivative of each ranked parameter w.r.t the fiducial values 
        dDW_dP[iteration], dSW_dP[iteration] = differentiation(ranks, 
                                                               N_data_points, 
                                                               N_sample, 
                                                               SNR_A, 
                                                               SNR_M,
                                                               z, 
                                                               R, 
                                                               model, 
                                                               M_qso_fiducial[0], 
                                                               M_qso_fiducial[1], 
                                                               seed, 
                                                               noise)
        iteration +=1

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

    mean_sampled_e_tau_1, _, mean_sampled_delta_SW_1, _ = sampler(noisy_e_tau_1, N_data_points, N_sample, seed)
    mean_sampled_e_tau_2, _, mean_sampled_delta_SW_2, _ = sampler(noisy_e_tau_2, N_data_points, N_sample, seed)

    delta_h =  np.log(M_qso_masses[1]) - np.log(M_qso_masses[0])

    dDW_dP[-1] = (mean_sampled_e_tau_2 - mean_sampled_e_tau_1)/delta_h
    dSW_dP[-1]= (mean_sampled_delta_SW_2 - mean_sampled_delta_SW_1)/delta_h

    # Getting the Fisher matrices
    # Case 1.1: uncorrelated pixel error Fisher Matrix:
    uncorr_fisher_matrix_DW, uncorr_cov_fisher_matrix_DW = compute_uncorr_fisher_matrix(var_sampled_e_tau_fiducial, dDW_dP, len(fisher_parameters.keys()))
    uncorr_fisher_matrix_SW, uncorr_cov_fisher_matrix_SW = compute_uncorr_fisher_matrix(var_sampled_delta_SW_fiducial, dSW_dP, len(fisher_parameters.keys()))

    # Case 1.2: correlated pixel error Fisher Matrix:
    corr_fisher_matrix_DW, corr_cov_fisher_matrix_DW = compute_corr_fisher_matrix(sampled_median_data, dDW_dP, len(fisher_parameters.keys()))
    corr_fisher_matrix_SW, corr_cov_fisher_matrix_SW = compute_corr_fisher_matrix(sampled_scatter_width, dSW_dP, len(fisher_parameters.keys()))

    # Case 1.3: combined fisher matrix for DW and SW with correlated pixel errors:
    combined_sampled_data = np.concatenate((sampled_median_data,sampled_scatter_width), axis=1)
    combined_ddata_dtheta = np.concatenate((dDW_dP,dSW_dP), axis=1)

    combined_corr_fisher_matrix, combined_corr_cov_fisher_matrix = compute_corr_fisher_matrix(combined_sampled_data, combined_ddata_dtheta, len(fisher_parameters.keys()))

    mean_values = np.array([params['target_xh'], 
                            round(np.log10(model.rank_list[0][-1]/365.25/86400),2), 
                            params['M_min'],
                            np.log10(M_qso_fiducial[0]*10**(M_qso_fiducial[1]-1))])        # Change the mass_bin number here as well when changing z
    
    combined_sample = np.random.multivariate_normal(mean_values, combined_corr_cov_fisher_matrix, size=10000) # combined fisher matrix for DW and SW with correlated pixel errors:
    DW_sample = np.random.multivariate_normal(mean_values, corr_cov_fisher_matrix_DW, size=10000)
    SW_sample = np.random.multivariate_normal(mean_values, corr_cov_fisher_matrix_SW, size=10000)
    
    fig_DW = plotting_fisher_matrix(DW_sample)
    axes = np.array(fig_DW.axes).reshape(len(fisher_parameters.keys()), len(fisher_parameters.keys()))  # Adjust for the number of parameters
    axes[0,2].set_title(f"Damping Wing SNR_A {SNR_A} SNR_M {SNR_M}")
    plt.savefig(f'{plotpath}/DW_contour_plot_SNR_A_{SNR_A}SNR_M_{SNR_M}_N_{N_data_points}_N_sample_{N_sample}.png')
    plt.close()
    
    fig_SW = plotting_fisher_matrix(SW_sample)
    axes = np.array(fig_SW.axes).reshape(4, 4)  # Adjust for the number of parameters
    axes[0,2].set_title(f"Scatter Width SNR_A {SNR_A} SNR_M {SNR_M}")
    plt.savefig(f'{plotpath}/SW_contour_plot_SNR_A_{SNR_A}SNR_M_{SNR_M}_N_{N_data_points}_N_sample_{N_sample}.png')
    plt.close()
    
    fig_combined = plotting_fisher_matrix(combined_sample)
    axes = np.array(fig_combined.axes).reshape(4, 4)  # Adjust for the number of parameters
    axes[0,2].set_title(f"Combined SNR_A {SNR_A} SNR_M {SNR_M}")
    plt.savefig(f'{plotpath}/Combined_contour_plot_SNR_A_{SNR_A}SNR_M_{SNR_M}_N_{N_data_points}_N_sample_{N_sample}.png')
    plt.close()

    return corr_cov_fisher_matrix_DW, corr_cov_fisher_matrix_SW, combined_corr_fisher_matrix, combined_corr_cov_fisher_matrix