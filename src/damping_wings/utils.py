"""
Utility functions shared across the damping wings pipeline.
Includes cosmological calculations used by multiple modules.
"""
import numpy as np
from numba import jit
import os

from .config import constants as _constants

@jit(nopython=True)
def H(z: float) -> float:
    """Hubble-Lemaître parameter at redshift z."""
    return _constants.H0 * (_constants.Omega_m*(1+z)**3 + _constants.Omega_lambda + _constants.Omega_k*((1+z)**2))**0.5

# @jit(nopython=True)
def calculate_t_vir (z: float, xh: float, m: float) -> float:
    """Calculates the Virial temperature for the given set of input variables

    Args:
        z (_type_): Redshift
        xh (_type_): Global mean neutral fraction
        m (_type_): Halo Mass

    Returns:
        float: Virial Temperature
    """
    Omega_m_z = (_constants.Omega_m*(1+z)**3)/(_constants.Omega_m*(1+z)**3 + _constants.Omega_lambda)                                 
    d = Omega_m_z**2 -1
    Delta_c = 18*np.pi**2 +82*d -39*d**2
    mu = xh*0.5 + (1-xh)
    t_vir = (1.98*10**4)*(mu/0.6)*((10**(m)*_constants.h/10**8)**(2/3))*(_constants.Omega_m*Delta_c/(Omega_m_z*18*np.pi**2))*((1+z)/10)
    return round(np.log10(t_vir), 2)

def setup_output_dirs() -> None:
    """Create required output directories. Call once before running the pipeline."""
    for path in [_constants.newpath, _constants.plotpath,
                 _constants.txt_files, _constants.cache_path]:
        os.makedirs(path, exist_ok=True)
