# src/damping_wings/config/__init__.py
"""
Configuration subpackage for damping_wings.
Contains physical constants and fiducial simulation parameters.

Users can modify constants and parameters before running the pipeline:

    from damping_wings.config import constants, parameters_file
    parameters_file.Parameters['z'] = 6.0
    constants.newpath = '/my/output/path'

Or import specific values directly:

    from damping_wings.config.constants import L_Box, HII_DIM
    from damping_wings.config.parameters_file import Parameters
"""

from . import constants
from . import parameters_file
from .parameters_file import Parameters
from .constants import (
    newpath, plotpath, txt_files, cache_path,
    L_Box, HII_DIM, DIM, N_sightlines, seed,
    SimParams, SimParamsRanges
)

__all__ = [
    "constants",
    "parameters_file",
    "Parameters",
    "newpath",
    "plotpath",
    "txt_files",
    "cache_path",
    "L_Box",
    "HII_DIM",
    "DIM",
    "N_sightlines",
    "seed",
    "SimParams",
    "SimParamsRanges",
]