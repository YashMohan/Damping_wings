"""
Modelling class for the damping wings pipeline.
Defines the Models class which orchestrates the full simulation suite.
"""
from typing import Optional
from numpy.typing import NDArray
import numpy as np
import sys
from ordered_set import OrderedSet
import itertools
import pickle
from multiprocessing import Process
from tabulate import tabulate
import time
import h5py
from scipy import optimize
import py21cmfast as p21c

#-----------------------------------------------------------------------------
# Models
from .config.constants import SimParams, SimParamsRanges, N_sightlines, newpath, H0, Omega_b, Omega_k, Omega_lambda, Omega_m, Nion, Conversion_amu_Mpc, L_Box, HII_DIM, DIM, txt_files, c, seed
from .config.parameters_file import Parameters as params 
from .ionized_boxes import generate_ion_boxes        # This code generates ionized boxes of the given parameters and initial conditions
from .calculating_skewers import calculate_skewers        # This code calculates the neutral fraction weighted over density from different halos along some random sightlines for a given ionized box
from .damping_wings import damping_wings        # For a given sightline, it calculates the damping wing profile for a specific halo mass host of a quasar
from .utils import calculate_t_vir
#-----------------------------------------------------------------------------

def Len(p: dict) -> int:
    """Returns the length of the input argument and returns 0 instead of none if the argument is an empty dictionary

    Args:
        p (dict): Input dictionary

    Returns:
        int: Length of the input dictionary
    """
    if p is None:
        return 0
    
    return len(p)

#-----------------------------------------------------------------------------
class Models():
    """ Generates a suite of simulation boxes for the given set of parameters and calculates the resultant Lyman alpha damping wing profiles
    """
    DD: int
    Parameters_names: list[str]
    Parameters_lens: list[int]
    Parameters_values: list[NDArray[np.float64]]
    Varying_order: list[int]
    rank_list: list[tuple[float, ...]]
    
    P_va: Optional[list[int]]
    P_vo: Optional[list[int]]
    n_va: Optional[list[NDArray[np.int64]]]
    n_vo: Optional[list[NDArray[np.int64]]]
    rank: Optional[list[int]]
    
    halo_mass: list[NDArray[np.float64]]
    halo_coords: list[NDArray[np.float64]]
    ionised_box: NDArray[np.float64]    # Neutral fraction of individual pixel of the simulation box
    density_field: NDArray[np.float64]  # Matter density of individual pixel of the simulation box
    
    results: Optional[NDArray[np.float64]]
    
    def __init__(self, param_ranges: SimParamsRanges) -> None:
    
        self.DD = len(params)    # Total number of parameters we have, -1 since we are not varying redshift
        self.Parameters_names = [k for k in param_ranges]    # Storing the names of the parameters
        self.Parameters_names = [k.replace('_list','') for k in self.Parameters_names] # removing _list from the Parameters names
        self.Parameters_values = [v for v in param_ranges.values()]       # Storing the range values in Parameters_values 
        self.Parameters_values = [np.insert(self.Parameters_values[index],0,params[f'{keys}']) for index,keys in enumerate(params)]    # Adding the fiducial values at front
        self.Parameters_values = [list(OrderedSet(i)) for i in self.Parameters_values]     # Making sure that the fiducial value appeaers only at front, hence ignoring if it repeats somewhere
        self.Parameters_lens = [len(i) for i in self.Parameters_values]   # length of each parameter, i.e., number of points for each parameter
        self.Varying_order = [np.prod(self.Parameters_lens[self.DD-1:i:-1]) for i in range(0,self.DD-1)] + [1]  # The point after which each parameter start to vary for the first time, 1 is added since the last element will vary immediately after the 0th (fiducial) model
        self.rank_list = list(itertools.product(*self.Parameters_values))   # List of all possible combinations of parameter values
        
        print(self.Parameters_names)
    
    def rank_calculation(
        self,
        Specific_values: Optional[dict[str, float]] = None,
        P_VA: Optional[list[str]] = None,
        P_VO: Optional[list[str]] = None,
        Avoid_mid: bool = False
        ) -> Optional[list[int]]:
        '''
        Determines the list of models for the given set of parameters by associating unique ranks to them
        
        Parameters
        ----------
        Specific_values : dictonary, optional
            Runs the model for a/some specific combination of values of some or all parameters. The default is None.
        P_VA : list, optional
            List of parameters which will vary for all the models (general form of corner models). The default is None.
        P_VO : list, optional
            List of parameters which will vary only once (for all values) throughout the run of all models (general form of face models). The default is None.
        Avoid_mid : bool, optional
            This flag will check if the user wants to vary the models over fiducial parameter values. 
            For example, we are varying our models for z = (7,6,8) where z = 7 is the fiducial value, if it is "True" then the model will run for all values of z, i.e., 7, 6 and 8,
            if it is "False" then the code will run only for z = 6 and 8, and avoid fiducial value, i.e., z = 7.
            The default is False.

        Returns
        -------
        TYPE: list
            Returns the list of ranks for the given set of arguments

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
            Paramms = dict(params)  # Using rest of the values from fiducial model
            Paramms.update(Specific_values)   # Updating the specific values on fiducial model
            indicies = [self.Parameters_values[i].index(Paramms[keys]) for i,keys in enumerate(Paramms)]  # Picking up the indicies of each parameter to calculate the rank corresponding to the specific parameters combo
            self.rank = [sum(j*self.Varying_order[i] for i,j in enumerate(indicies))]   # Getting the rank for the specific parameters combo
        
            return self.rank
        
        if P_VA is not None:
            P_VA = set(P_VA)
            self.P_va = [self.Parameters_names.index(j) for j in P_VA]  # If P_VA is provided, then storing the position of the provided parameters in P_va
            self.n_va = [np.linspace(0,self.Parameters_lens[i]-1,self.Parameters_lens[i], dtype=int) for i in self.P_va] # Storing the length or number of points for P_VA parameter list in n_va
            if Avoid_mid:
                self.n_va = [np.linspace(1,self.Parameters_lens[i]-1,self.Parameters_lens[i]-1, dtype=int) for i in self.P_va] # Storing the length or number of points for P_VA parameter list in n_va
                
        if P_VO is not None:
            self.P_vo = [self.Parameters_names.index(j) for j in P_VO]  # If P_VO is provided, then storing the position of the provided parameters in P_vo
            self.n_vo = [np.linspace(0,self.Parameters_lens[i]-1,self.Parameters_lens[i], dtype=int) for i in self.P_vo]  # Storing the length or number of points for P_VO parameter list in n_vo
            if Avoid_mid:
                self.n_vo = [np.linspace(1,self.Parameters_lens[i]-1,self.Parameters_lens[i]-1, dtype=int) for i in self.P_vo] # Storing the length or number of points for P_VA parameter list in n_va
              
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
    
    def halo_data(self,mass_index: int) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int, int]:
        """ Returns the list of halos with the desired mass and their coordinates

        Args:
            mass_index (int): Index to select the desired mass from the set of halo masses generated by the simulation box

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], int, int, int]: Returns the desired halo mass and the coordinates, number of halos, simplified order and base for the desired halo mass
        """
        
        m: tuple[bool]  # Selecting only the halos of the desired mass
        new_halo_mass: NDArray[np.float64]   # Storing all those halos with the desired halo mass as new_halo_mass
        new_halo_coords: NDArray[np.float64]    # Storing their corresponding coordinates in new_halo_coords
        n_halos: int    # Number of halos in the selected mass bin
        o_halo_mass: int    # The order of the mass of the halo in the form of Mass = base*10^order, this is an approximation not the exact value of the masses
        base_halo_mass: int # The base from Mass = base*10^order, base and order are used to separate halos of different masses                
        
        Mass_bins = np.unique(self.halo_mass)
        m = (self.halo_mass == Mass_bins[mass_index])     
        new_halo_mass = self.halo_mass[m]        
        new_halo_coords = self.halo_coords[m]    
        n_halos = len(new_halo_mass)
        print(f"Unique masses in bin {mass_index}: {np.unique(new_halo_mass)}")
        print(f"Number of unique: {len(np.unique(new_halo_mass))}")
        o_halo_mass = int(np.floor(np.log10(np.unique(new_halo_mass))))             
        base_halo_mass = int(np.round(new_halo_mass[0]/(10**o_halo_mass),0)) 
        
        return new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass 
    
    def damping_wings_calculations(self, mass_index: int, Parameters: SimParams, rank: int) -> None:
        '''
        Calculates the ensemble of Lyman alpha damping wing profiles for the desired halo mass, and astrophysical parameters determined by the unique rank

        Parameters
        ----------
        mass_index : int
            Index to select the desired mass from the set of halo masses generated by the simulation box.
        Parameters : SimParams
            Set of astrophysical parameters governing the simulation
        rank : int
            Unique rank identifier for the given set of astrophysical parameters

        Returns
        -------
        None.

        '''
        new_halo_mass: NDArray[np.float64]   # Storing all those halos with the desired halo mass as new_halo_mass
        new_halo_coords: NDArray[np.float64]    # Storing their corresponding coordinates in new_halo_coords
        n_halos: int    # Number of halos in the selected mass bin
        o_halo_mass: int    # The order of the mass of the halo in the form of Mass = base*10^order, this is an approximation not the exact value of the masses
        base_halo_mass: int # The base from Mass = base*10^order, base and order are used to separate halos of different masses 
        
        new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass = self.halo_data(mass_index)
        
        calculate_skewers(base_halo_mass, o_halo_mass, new_halo_coords, self.ionised_box, self.density_field, Parameters, rank)
        print("Calculating the ensemble of Lyman-alpha damping wing profiles")
        damping_wings(base_halo_mass, o_halo_mass, Parameters, rank) 

    
    def modelling(self, 
                  initial_conditions: p21c.InitialConditions,
                  cache_obj: p21c.OutputCache,
                  rank: Optional[int] = None,
                  Out_of_bound_parameters: Optional[SimParams] = None,
                  seed: Optional[int] = 54321
                  ) -> None:
        '''
        For a given set of initial conditions and astrophysical parameters, generates the simulation box and calculates the ensemble of damping wing profiles
        
        Parameters
        ----------
        initial_conditions: p21c.InitialConditions
            Initial conditions for the 21cmFAST simulation box
        cache_obj: p21c.OutputCache
            Cache object and path to store the cache files from the simulation
        rank : list, optional
            List of ranks over which we will calculate our models. The default is None, which corresponds to thee fiducial model.
        Out_of_bound_parameters
            To run models outside of the range of given parameters space
        seed : Optional[int]
            Seed for the simulation box
        Returns
        -------
        None.
        
        '''
        
        if Out_of_bound_parameters is not None:
            
            rank_index = -1
            
            Parameters = dict(params)  # Using rest of the values from fiducial model
            Parameters.update(Out_of_bound_parameters)
            Parameters = dict(Out_of_bound_parameters)
            Parameters['T_vir'] = calculate_t_vir(z = Parameters['z'], xh = Parameters['target_xh'], m = Parameters['m_min'])
            print(Parameters)
            generate_ion_boxes(initial_conditions,cache_obj,Parameters,rank_index)   # Generating the ionized box
            
            #--------------------------------------------------------------------------------------------------------------------------------------------------------
            # Skewers calculations
            print("\nLoading the halo files and the ionized box")
            # Loading all the halos with their masses and coords, and the ionized and desnity of the box
            self.halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
            self.halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
            self.ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
            self.density_field = pickle.load( open(f"{newpath}/Density_field_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
            
            #Picking up all the masss bins
            mass_bins = np.unique(self.halo_mass)
            n_mass_bins = len(mass_bins)
            print("\nMass bins: ", mass_bins)
            
            with open(f'{txt_files}/Additional_data_{rank_index}_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt', 'a') as f:
                f.write(f'Mass bins {mass_bins} \n')
                
                
            halo_data_columns = ["Base", "Order", "No. of halos", "Halo Mass"]
            halo_opted = []
            
            with open(f"{txt_files}/Halos_for_skewers_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt", 'w') as f:
                for i in range(0,n_mass_bins,int(n_mass_bins/5)):
                    new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass = self.halo_data(i)
                    halo_opted.append([base_halo_mass, o_halo_mass, n_halos, np.log10(new_halo_mass[0])])
                    f.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
                
                if (n_mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
                    i = n_mass_bins-1     
                    new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass = self.halo_data(i)
                    halo_opted.append([base_halo_mass, o_halo_mass, n_halos, np.log10(new_halo_mass[0])])
                    f.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
            
            print("\n",tabulate(halo_opted, headers=halo_data_columns))     
                
            process = []
            print("\nCalculating the sightlines")
            start = time.perf_counter()
            for i in range(0,n_mass_bins,int(n_mass_bins/5)):
                # create a process
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,rank_index))
                p.start()
                process.append(p)
            
            if (n_mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
                i = n_mass_bins-1    
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,rank_index))
                p.start()
                process.append(p)
                
            for p in process:
                p.join()
            
            self.damping_wings_calculations(-1, Parameters, rank_index)
            end = time.perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))
            
            return 
            
    
        if rank is None:
            rank = [0]
            
        self.rank = rank
        
        for rank_index in self.rank:
        
            # Calling the Generate Ionized box function 
            Parameters = dict(zip(self.Parameters_names,self.rank_list[rank_index]))
            Parameters['T_vir'] = calculate_t_vir(z = Parameters['z'], xh = Parameters['target_xh'], m = Parameters['m_min'])
            print(Parameters)
            print("seed: ",seed," L_Box: ",L_Box)
            generate_ion_boxes(initial_conditions,cache_obj,Parameters,rank_index)   # Generating the ionized box
            
            #--------------------------------------------------------------------------------------------------------------------------------------------------------
            # Skewers calculations
            print("\nLoading the halo files and the ionized box")
            # Loading all the halos with their masses and coords, and the ionized and desnity of the box
            self.halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
            self.halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
            self.ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
            self.density_field = pickle.load( open(f"{newpath}/Density_field_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
            
            #Picking up all the masss bins
            mass_bins = np.unique(self.halo_mass)
            n_mass_bins = len(mass_bins)
            print("\nMass bins: ", mass_bins)

            # breakpoint()
            
            with open(f'{txt_files}/Additional_data_{rank_index}_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt', 'a') as f:
                f.write(f'Mass bins {mass_bins} \n')     
                
            halo_data_columns = ["Base", "Order", "No. of halos", "Halo Mass"]
            halo_opted = []
            
            # with open(f"{txt_files}/Halos_for_skewers_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt", 'w') as f:
            with open(f"{txt_files}/Halos_for_skewers_rank_{rank_index}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.txt", 'w') as f:
                for i in range(0,n_mass_bins,int(n_mass_bins/5)):
                    new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass = self.halo_data(i)
                    halo_opted.append([base_halo_mass, o_halo_mass, n_halos, np.log10(new_halo_mass[0])])
                    f.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
                
                if (n_mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
                    i = n_mass_bins-1     
                    new_halo_mass, new_halo_coords, n_halos, o_halo_mass, base_halo_mass = self.halo_data(i)
                    halo_opted.append([base_halo_mass, o_halo_mass, n_halos, np.log10(new_halo_mass[0])])
                    f.write(f"{base_halo_mass} {o_halo_mass} {'{:.3e}'.format(n_halos)} {'{:.2f}'.format(np.log10(new_halo_mass[0]))}\n")
            
            print("\n",tabulate(halo_opted, headers=halo_data_columns))
            
            process = []
            print("\nCalculating the sightlines")
            start = time.perf_counter()
            for i in range(0,n_mass_bins,int(n_mass_bins/5)):
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,rank_index))
                p.start()
                process.append(p)
            
            if (n_mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
                i = n_mass_bins-1    
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,rank_index))
                p.start()
                process.append(p)
                
            for p in process:
                p.join()
                
            end = time.perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))
            
            
    def calibrate_function(self, calibrate_value: float,
                           calibrate_variable: str,
                           e_tau_avg_expected: NDArray[np.float64],
                           initial_conditions: p21c.InitialConditions, 
                           cache_obj: p21c.OutputCache) -> float:
        '''
        Objective function for brenth calibration. Returns normalised mean residual between expected and predicted transmission profiles.
        Parameters
        ----------
        calibrate_value : float
            Desired value of the variable which needs to be calibrated
        calibrate_variable : str
            Variable needs to be calibrated
        e_tau_avg_expected : NDArray[np.float64]
            Expected mean transmission profile
        initial_conditions: p21c.InitialConditions
            Initial conditions for the 21cmFAST simulation box
        cache_obj: p21c.OutputCache
            Cache object and path to store the cache files from the simulation

        Returns
        -------
        normalised_residual : float
            Returns the mean residual between expected and predicted transmission profiles.

        '''
        calibrating_parameters = dict(params) 
        new_values = {calibrate_variable:calibrate_value}
        calibrating_parameters.update(new_values)
        
        print(calibrating_parameters)
        
        self.modelling(initial_conditions,cache_obj, Out_of_bound_parameters = calibrating_parameters)
        
        base = []
        order = []
        num_halos = []
        mass_halos = []
        with open(f"{txt_files}/Halos_for_skewers_rank_-1_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt", 'r') as file:
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
        
        with h5py.File(f"{newpath}/skewers_HM_{base[-1]}_{order[-1]}_rank_-1_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.h5", 'r') as f:
            lamda = f.get("lambda")[:]
            e_tau_avg_pred = f.get("e_tau_avg")[:]
            
        mask: NDArray[np.bool_] = e_tau_avg_expected > 1e-6
        normalised_residual = np.mean(
            (e_tau_avg_expected[mask] - e_tau_avg_pred[mask]) / e_tau_avg_expected[mask])
        
        return normalised_residual
            
            
    def calibrating_variables(self, calibration_variable: str,
                              target_variable: dict[str, float],
                              initial_conditions: p21c.InitialConditions,
                              cache_obj: p21c.OutputCache) -> float:
        '''
        

        Parameters
        ----------
        calibrate_variable : str
            Variable needs to be calibrated
        target_variable: dict[str, float]
            Target variable for calibration
        initial_conditions: p21c.InitialConditions
            Initial conditions for the 21cmFAST simulation box
        cache_obj: p21c.OutputCache
            Cache object and path to store the cache files from the simulation

        Returns
        -------
        calibrate_value : float
            Calibrated value of the calibrate_variable to match the transmission profile from target_variable

        '''
        
        Parameters = dict(params)  # Using rest of the values from fiducial model
        Parameters.update(target_variable)
        
        rank: list = self.rank_calculation(Specific_values = target_variable)
        
        self.modelling(initial_conditions, cache_obj, rank=rank)
        
        base: list[int] = []
        order: list[int] = []
        num_halos: list[str] = []
        mass_halos: list[str] = []
        lamda: NDArray[np.float64]
        e_tau_avg_expected: NDArray[np.float64]
        lower_end: float
        upper_end: float
        
        with open(f"{txt_files}/Halos_for_skewers_rank_{rank[0]}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt", 'r') as file:
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
        
        with h5py.File(f"{newpath}/skewers_HM_{base[-1]}_{order[-1]}_rank_{rank[0]}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.h5", 'r') as f:
            lamda = f.get("lambda")[:]
            e_tau_avg_expected = f.get("e_tau_avg")[:]
        
        
        lower_end = self.Parameters_values[self.Parameters_names.index(calibration_variable)][1]
        upper_end = self.Parameters_values[self.Parameters_names.index(calibration_variable)][-1]
        
        calibrate_value = optimize.brenth(self.calibrate_function,lower_end, upper_end, args=(calibration_variable, e_tau_avg_expected, initial_conditions, cache_obj))
        
        return calibrate_value

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
