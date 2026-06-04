# tests/test_config.py
from damping_wings.config import constants, Parameters

def test_constants_have_required_attributes():
    """All required constants should be present"""
    required = ['H0', 'Omega_m', 'Omega_lambda', 'Omega_b', 'c', 'G',
                'L_Box', 'HII_DIM', 'DIM', 'N_sightlines', 'seed',
                'newpath', 'plotpath', 'txt_files', 'cache_path']
    for attr in required:
        assert hasattr(constants, attr), f"Missing constant: {attr}"

def test_constants_physical_values():
    """Physical constants should be in reasonable ranges"""
    assert 60000 < constants.H0 < 80000      # m/s/Mpc
    assert 0 < constants.Omega_m < 1
    assert 0 < constants.Omega_lambda < 1
    assert abs(constants.Omega_m + constants.Omega_lambda - 1.0) < 0.01

def test_parameters_have_required_keys():
    """Fiducial parameters dict should have all required keys"""
    required = ['z', 'M_min', 'target_xh', 'alpha_esc',
                'alpha_star', 'f_star', 'tq']
    for key in required:
        assert key in Parameters, f"Missing parameter: {key}"

def test_parameters_physical_ranges():
    """Fiducial parameters should be in physical ranges"""
    assert 5 < Parameters['z'] < 15
    assert 0 < Parameters['target_xh'] < 1