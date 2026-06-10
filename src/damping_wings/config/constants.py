import datetime
import os
from typing import TypedDict
from numpy.typing import NDArray
import numpy as np

today = str(datetime.date.today())
#-----------------------------------------------------------------------------
newpath: str = os.environ.get(
    'DAMPING_WINGS_OUTPUT',
    os.path.join(os.getcwd(), 'output')
)

plotpath: str  = os.path.join(newpath, 'plots')
txt_files: str = os.path.join(newpath, 'txt_files')
cache_path: str = os.path.join(newpath, 'cache_files')
#-----------------------------------------------------------------------------
class SimParams(TypedDict):
    z: float
    m_min: float
    target_xh: float
    alpha_esc: float
    alpha_star: float
    f_star: float
    tq: float

class SimParamsRanges(TypedDict):
    x_hi: NDArray[np.float64]
    m_min: NDArray[np.float64]
    t_q: NDArray[np.float64]
    m_qso: NDArray[np.float64]
    redshift: NDArray[np.float64]
#-----------------------------------------------------------------------------
    
#Constants

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


Nion = 10**56   # Number of ionising photons
Nion = np.float64(Nion)

Lambda = 6.25*10**8 #s^-1
nu_alpha = 2.47*10**15  # hz
R_alpha = Lambda/(4*np.pi*nu_alpha)

L_Box = 25
DIM = 100
HII_DIM = 25

dl = L_Box/HII_DIM      # Differential step in box
n_pixels = int(300*HII_DIM/L_Box)     # Making sure that the distance travelled by the sightline is always the same

No_of_halos = 100

N_sightlines = 10000 # Number of sightlines

seed = 54321
#-----------------------------------------------------------------------------
