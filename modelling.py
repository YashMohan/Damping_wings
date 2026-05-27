"""
Created on Mon Mar 20 15:12:17 2023

@author: sharma
"""
import numpy as np
import sys
import logging
from ordered_set import OrderedSet
import itertools
import pickle
from multiprocessing import Process
from tabulate import tabulate
import time
import h5py
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import qmc
import py21cmfast as p21c

#-----------------------------------------------------------------------------
# Models
from constants import *
import m_pixels as mp
import parameters_file as params
import ionized_boxes as ib      # This code generates ionized boxes of the given parameters and initial conditions
# mport calculating_skewers as cs        # This code calculates the neutral fraction weighted over density from different halos along some random sightlines for a given ionized box
# import damping_wings as dw         # For a given sightline, it calculates the damping wing profile for a specific halo mass host of a quasar
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

def Len(p):
    if p is None:
        return 0
    
    return len(p)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def calculate_t_vir (z,xh,m):
    Omega_m_z = (Omega_m*(1+z)**3)/(Omega_m*(1+z)**3 + Omega_lambda)                                 
    d = Omega_m_z**2 -1
    Delta_c = 18*np.pi**2 +82*d -39*d**2
    mu = xh*0.5 + (1-xh)
    t_vir = (1.98*10**4)*(mu/0.6)*((10**(m)*h/10**8)**(2/3))*(Omega_m*Delta_c/(Omega_m_z*18*np.pi**2))*((1+z)/10)
    t_vir =  float('{:.2f}'.format(np.log10(t_vir)))
    return t_vir

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
class Models():
    
    def __init__(self, param_ranges):
    
        self.DD = len(params.Parameters)    # Total number of parameters we have, -1 since we are not varying redshift
        self.Parameters_names = [k for k in param_ranges]    # Storing the names of the parameters
        self.Parameters_names = [k.replace('_list','') for k in self.Parameters_names] # removing _list from the Parameters names
        self.Parameters_values = [v for v in param_ranges.values()]       # Storing the range values in Parameters_values 
        self.Parameters_values = [np.insert(self.Parameters_values[index],0,params.Parameters[f'{keys}']) for index,keys in enumerate(params.Parameters)]    # Adding the fiducial values at front
        self.Parameters_values = [list(OrderedSet(i)) for i in self.Parameters_values]     # Making sure that the fiducial value appeaers only at front, hence ignoring if it repeats somewhere
        self.Parameters_lens = [len(i) for i in self.Parameters_values]   # length of each parameter, i.e., number of points for each parameter
        self.Varying_order = [np.prod(self.Parameters_lens[self.DD-1:i:-1]) for i in range(0,self.DD-1)] + [1]  # The point after which each parameter start to vary for the first time, 1 is added since the last element will vary immedietly after the 0th (fiducial) model
        self.rank_list = list(itertools.product(*self.Parameters_values))   # List of all possible combinations of parameter values
        
        print(self.Parameters_names)
    
    def rank_calculation(self, Specific_values = None, P_VA = None, P_VO = None, Avoid_mid = False):
        '''
        

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
            Paramms = dict(params.Parameters)  # Using rest of the values from fiducial model
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
    
    def halo_data(self,i):
        '''
        

        Parameters
        ----------
        i : TYPE
            Mass bin

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
        print(base_halo_mass, o_halo_mass, np.shape(new_halo_coords), np.shape(self.ionised_box), np.shape(self.density_field), I, "END")
        cs.calculate_skewers(base_halo_mass, o_halo_mass, new_halo_coords, self.ionised_box, self.density_field, Parameters, I)
        print("Damping wing function started")
        dw.damping_wings(base_halo_mass, o_halo_mass, Parameters, I)  # Calculating damping wings for different halo mass

    
    def modelling(self, initial_conditions,cache_obj, rank = None, Out_of_bound_parameters = None, seed = 54321):
        '''
        
        
        Parameters
        ----------
        rank : list, optional
        List of ranks over which we will calculate our models. The default is None, which corresponds to thee fiducial model.
        
        Returns
        -------
        None.
        
        '''
        
        if Out_of_bound_parameters is not None:
            
            I = -1
            
            Parameters = dict(params.Parameters)  # Using rest of the values from fiducial model
            Parameters.update(Out_of_bound_parameters)
            Parameters = dict(Out_of_bound_parameters)
            Parameters['T_vir'] = calculate_t_vir(z = Parameters['z'], xh = Parameters['target_xh'], m = Parameters['m_min'])
            print(Parameters)
            ib.generate_ion_boxes(initial_conditions,cache_obj,Parameters,I)   # Generating the ionized box
            
            #--------------------------------------------------------------------------------------------------------------------------------------------------------
            # Skewers calculations
            print("\nLoading the halo files and the ionized box")
            # Loading all the halos with their masses and coords, and the ionized and desnity of the box
            self.halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
            self.halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
            self.ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
            self.density_field = pickle.load( open(f"{newpath}/Density_field_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
            
            #Picking up all the masss bins
            mass_bins = np.unique(self.halo_mass)
            n_mass_bins = len(mass_bins)
            print("\nMass bins: ", mass_bins)
            
            with open(f'{txt_files}/Additional_data_{I}_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt', 'a') as f:
                f.write(f'Mass bins {mass_bins} \n')
                
                
            halo_data_columns = ["Base", "Order", "No. of halos", "Halo Mass"]
            halo_opted = []
            
            with open(f"{txt_files}/Halos_for_skewers_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt", 'w') as f:
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
                
            # process = []
            print("\nCalculating the sightlines")
            start = time.perf_counter()
            for i in range(0,n_mass_bins,int(n_mass_bins/5)):
                # create a process
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,I))
                p.start()
                process.append(p)
            
            if (n_mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
                i = n_mass_bins-1    
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,I))
                p.start()
                process.append(p)
                
            for p in process:
                p.join()
            
            self.damping_wings_calculations(-1, Parameters, I)
            end = time.perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))
            
            return 
            
    
        if rank is None:
            rank = [0]
            
        self.rank = rank
        
        for I in self.rank:
        
            # Calling the Generate Ionized box function 
            Parameters = dict(zip(self.Parameters_names,self.rank_list[I]))
            Parameters['T_vir'] = calculate_t_vir(z = Parameters['z'], xh = Parameters['target_xh'], m = Parameters['m_min'])
            print(Parameters)
            print("seed: ",seed," L_Box: ",L_Box)
            ib.generate_ion_boxes(initial_conditions,cache_obj,Parameters,I)   # Generating the ionized box
            
            #--------------------------------------------------------------------------------------------------------------------------------------------------------
            # Skewers calculations
            print("\nLoading the halo files and the ionized box")
            # Loading all the halos with their masses and coords, and the ionized and desnity of the box
            self.halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p","rb"))
            self.halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p","rb"))
            self.ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p", "rb" ))
            self.density_field = pickle.load( open(f"{newpath}/Density_field_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.p", "rb" ))
            # self.halo_mass = pickle.load(open(f"{newpath}/Halo_masses_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
            # self.halo_coords = pickle.load(open(f"{newpath}/Halo_coords_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p","rb"))
            # self.ionised_box = pickle.load( open(f"{newpath}/Ionized_box_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
            # self.density_field = pickle.load( open(f"{newpath}/Density_field_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.p", "rb" ))
            
            #Picking up all the masss bins
            mass_bins = np.unique(self.halo_mass)
            n_mass_bins = len(mass_bins)
            print("\nMass bins: ", mass_bins)
            
            # with open(f'{txt_files}/Additional_data_{I}_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt', 'a') as f:
            with open(f'{txt_files}/Additional_data_{I}_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.txt', 'a') as f:
                f.write(f'Mass bins {mass_bins} \n')     
                
            halo_data_columns = ["Base", "Order", "No. of halos", "Halo Mass"]
            halo_opted = []
            
            # with open(f"{txt_files}/Halos_for_skewers_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt", 'w') as f:
            with open(f"{txt_files}/Halos_for_skewers_rank_{I}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}.txt", 'w') as f:
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
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,I))
                p.start()
                process.append(p)
            
            if (n_mass_bins-1)%5:       # In case the (number of bins-1) is not the multiple of 5, then the last/most massive halo is skipped from the above loop, this section makes sure to always include the most massive halo in the picture
                i = n_mass_bins-1    
                p = Process(target= self.damping_wings_calculations, args=(i, Parameters,I))
                p.start()
                process.append(p)
                
            for p in process:
                p.join()
                
            end = time.perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))
            
            
    def calibrate_function(self, calibrate_value, calibrate_variable, e_tau_avg_expected, initial_conditions):
        '''
        

        Parameters
        ----------
        calibrate_value : TYPE
            DESCRIPTION.
        calibrate_variable : TYPE
            DESCRIPTION.
        Parameters : TYPE
            DESCRIPTION.
        e_tau_avg_expected : TYPE
            DESCRIPTION.
        initial_conditions : TYPE
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        '''
        calibrating_parameters = dict(params.Parameters) 
        new_values = {calibrate_variable:calibrate_value}
        calibrating_parameters.update(new_values)
        
        print(calibrating_parameters)
        
        self.modelling(initial_conditions, Out_of_bound_parameters = calibrating_parameters)
        
        base = []
        order = []
        num_halos = []
        mass_halos = []
        file = open(f"{txt_files}/Halos_for_skewers_rank_-1_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt",'r')
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
            
        # chi_2 = chisquare(e_tau_avg_pred, e_tau_avg_expected)
        chi_2 = 100*sum(e_tau_avg_expected - e_tau_avg_pred)/len(e_tau_avg_expected)
        # print("chi squared = ",chi_2)
        
        # result = 100*sqrt(mean_squared_error(e_tau_avg_expected, e_tau_avg_pred)) 
        
        # result = 100*sum(e_tau_avg_expected - e_tau_avg_pred)/len(e_tau_avg_expected)
        
        return chi_2
            
            
    def calibrating_variables(self, calibration_variable, target_variable, initial_conditions):
        '''
        

        Parameters
        ----------
        calibration_variable : TYPE
            DESCRIPTION.
        target_variable : TYPE
            DESCRIPTION.
        initial_conditions : TYPE
            DESCRIPTION.

        Returns
        -------
        calibrate_value : TYPE
            DESCRIPTION.

        '''
        
        Parameters = dict(params.Parameters)  # Using rest of the values from fiducial model
        Parameters.update(target_variable)
        
        rank = self.rank_calculation(Specific_values = target_variable)
        
        self.modelling(initial_conditions, rank)
        
        base = []
        order = []
        num_halos = []
        mass_halos = []
        file = open(f"{txt_files}/Halos_for_skewers_rank_{rank[0]}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.txt",'r')
        for l in file.readlines():
            b, o, n, m = l.strip().split(" ")
            base.append(int(b))
            order.append(int(o))
            num_halos.append(n)
            mass_halos.append(m)
            
        base = np.array(base)
        order = np.array(order)
        num_halos = np.array(num_halos)#YMjkS19697867
        
        mass_halos = np.array(mass_halos)
        
        with h5py.File(f"{newpath}/skewers_HM_{base[-1]}_{order[-1]}_rank_{rank[0]}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.h5", 'r') as f:
            lamda = f.get("lambda")[:]
            e_tau_avg_expected = f.get("e_tau_avg")[:]
            
        # with h5py.File(f"{newpath}/quantile_data_{base[0]}_{order[0]}_rank_{rank[0]}_no_halofield_DIM_{DIM}_HII_{HII_DIM}_L_{L_Box}_N_{N_sightlines}_seed_{seed}.h5", 'r') as f:
        #     low_expected = f.get("low_quantile")[:]
        #     mid_expected = f.get("mid_quantile")[:]
        #     up_expected = f.get("up_quantile")[:]
        
        # plt.plot(lamda,e_tau_avg_expected)
        
        lower_end = self.Parameters_values[self.Parameters_names.index(calibration_variable)][1]
        upper_end = self.Parameters_values[self.Parameters_names.index(calibration_variable)][-1]
        
        # x0 = [lower_end]
        
        calibrate_value = optimize.brenth(self.calibrate_function,lower_end, upper_end, args=(calibration_variable, e_tau_avg_expected, initial_conditions))
        
        # calibrate_value = optimize.minimize(self.calibrate_function, x0, method='nelder-mead',
                # args=(calibration_variable, Parameters, e_tau_avg_expected, initial_conditions), options={'xatol': 1e-3, 'disp': True})
        
        print(calibrate_value)
        
        return calibrate_value

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #-----------------------------------------------------------------------------
    # Logger module
    # logger = logging.getLogger("Damping_wings")

    # logFileFormatter = logging.Formatter(
    #     fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )
    # fileHandler = logging.FileHandler(filename=f'{newpath}/logging.log')
    # fileHandler.setFormatter(logFileFormatter)
    # fileHandler.setLevel(level=logging.INFO)

    # logger.addHandler(fileHandler)

    #-----------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    start_time = time.perf_counter()
    m_p = mp.Get_me_M_min(initial_conditions)        # Average pixel mass; run once
    time_elapsed = time.perf_counter() - start_time
    # logger.info(f"Pixel mass = {M_p}")
    # logger.info(f"It took {time_elapsed} seconds to calculate the pixel mass")

    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------

    m_min = float('{:.2f}'.format(np.log10(m_p*20)))
    # m_min = 9.58
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
    sys.exit()

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

    r = functools.reduce(operator.iconcat, r, [])

    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    ## Section to calibrate with respect to one variable to find the effective value of the
    # mass_bins = np.unique(model.halo_mass)
    # n_mass_bins = len(mass_bins)

    # calibration_variable = 'target_xh'

    # target_variable = {'tq':params.Parameters['tq']}
    # target_variable = {'tq':0.0}
    
    # effective_xh = model.calibrating_variables(calibration_variable, target_variable, initial_conditions)
    
