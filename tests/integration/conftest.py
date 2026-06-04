"""
Shared fixtures for integration tests.
Requires 21cmFAST and sufficient RAM (~8GB for DIM=100).
Run with: pytest tests/integration/ -v
"""
import pytest
import py21cmfast as p21c
from damping_wings import setup_output_dirs
from damping_wings.config import constants
import tempfile
import os

@pytest.fixture(scope="session")
def tmp_output(tmp_path_factory):
    """Temporary output directory for the full test session."""
    path = tmp_path_factory.mktemp("damping_wings_test")
    constants.newpath   = str(path)
    constants.plotpath  = os.path.join(str(path), 'plots')
    constants.txt_files = os.path.join(str(path), 'txt_files')
    constants.cache_path= os.path.join(str(path), 'cache_files')
    setup_output_dirs()
    return path

@pytest.fixture(scope="session")
def cache_obj(tmp_output):
    """21cmFAST output cache pointed at temp directory."""
    return p21c.OutputCache(constants.cache_path)

@pytest.fixture(scope="session")
def initial_conditions(cache_obj):
    """
    Compute initial conditions once for the full test session.
    Uses minimal box size for speed.
    """
    new_inputs = p21c.InputParameters(
        simulation_options={
            "DIM": 100,
            "HII_DIM": 25,
            "BOX_LEN": 25
        },
        matter_options={
            "USE_FFTW_WISDOM": False,
            "PERTURB_ALGORITHM": "2LPT",
            "SOURCE_MODEL": "E-INTEGRAL"
        },
        astro_options={
            "M_MIN_in_Mass": True,
            "USE_EXP_FILTER": False,
            "USE_UPPER_STELLAR_TURNOVER": False
        },
        cosmo_params=p21c.CosmoParams(SIGMA_8=0.8, OMm=0.3, OMb=0.045),
        random_seed=constants.seed
    )
    return p21c.compute_initial_conditions(
        inputs=new_inputs,
        cache=cache_obj,
        write=True
    )