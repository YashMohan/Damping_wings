import numpy as np
import datetime
import os
from Parameters_file import Parameters 

today = str(datetime.date.today())
#-----------------------------------------------------------------------------
#For cluster
# newpath = r'/data/beegfs/astro-storage/groups/davies/sharma/21cmFast_work/'+today 
# if not os.path.exists(newpath):
#     os.mkdir(newpath)

# plotpath = newpath+'/Plots'
# if not os.path.exists(plotpath):
#     os.mkdir(plotpath)

#-----------------------------------------------------------------------------    
#For laptop
newpath = r'/Users/sharma/work/21cmFast_codes_and_plots/Modified_Calibration/'+today 
if not os.path.exists(newpath):
    os.mkdir(newpath)

#Plot path
plotpath = newpath+'/Plots'
if not os.path.exists(plotpath):
    os.mkdir(plotpath)
    
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
    
#Constants

n_pixels = 300
delta = 1 #differential interval to move in pixel space

H0 = 70000.0    # units: m/s/Mpc
Omega_m = 0.3
Omega_lambda = 0.7
Omega_k = 0.0
Omega_b = 0.045
c = 3*10**8 #ms^-1
G = 6.67*10**(-11) # units: m*((m/s)**2)/kg
h = 0.7

Conversion_amu_Mpc = (6.022/3.241)*10**49
Conversion_m_to_Mpc = 3.24*10**(-23)
Conversion_kg_Solar_mass = 5.027*10**(-31)


Nion = 10**57   # Number of ionising photons

Lambda = 6.25*10**8 #s^-1
nu_alpha = 2.47*10**15  # hz
R_alpha = Lambda/(4*np.pi*nu_alpha)


L_Box = 256 #Mpc Co-moving
DIM = 1024
HII_DIM = 256

dl = L_Box/HII_DIM      # Differential step in box
n_pixels = int(300*HII_DIM/L_Box)

N_sightlines = 100 # Number of sightlines
#-----------------------------------------------------------------------------
