"""
damping_wings — A general OOP pipeline for parametric semi-numerical
simulation and Fisher Information Matrix-based parameter inference.

Developed at the Max Planck Institute for Astronomy.
Author: Yash Mohan Sharma
"""

from .modelling_class import Models
from .utils import H, calculate_t_vir, setup_output_dirs

__version__ = "0.1.0"
__author__ = "Yash Mohan Sharma"
__email__ = "yashmohansharma96@gmail.com"

__all__ = [
    "Models",
    "H",
    "calculate_t_vir",
    "setup_output_dirs",
]