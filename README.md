# Damping_wings
This code generates the ionised boxes with given list of parameters and calculates the damping wing profiles, treating halos (random or fixed) as the source of quasars.

# Installation
To run this code, simply download the source code. Then change the "newpath" in "Constants.py" to your desired location.
Then run "Running_models_non_parallel_git.py"

# Pre-requisites
1. 21cmFast : https://github.com/21cmfast/21cmFAST
2. HMFcalc : https://github.com/halomod/HMFcalc
3. tqdm : pip install tqdm
4. tabulate ; pip install tabulate

# Changing the parameters
1. To change the parameters you want to vary or to change the range of the parameters, open and edit "Parameters_file.py". This file contains the general list of the parametrs.
2. To change the parameters and the ranges only for the corner models, edit the "Param_Ranges" dictionary in "Running_models_non_parallel_git.py". 
3. To change the parameters for face models, edit "variables_list" in "Running_models_non_parallel_git.py".
