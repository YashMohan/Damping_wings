# tests/test_modelling_class.py
import numpy as np
from damping_wings.modelling_class import Models, Len
from damping_wings.config import Parameters, constants
import tempfile, os

def test_Len_with_none():
    assert Len(None) == 0

def test_Len_with_empty_dict():
    assert Len({}) == 0

def test_Len_with_dict():
    assert Len({'a': 1, 'b': 2}) == 2

def test_models_init_parameter_names(tmp_path):
    """Models should correctly parse parameter names from param_ranges"""
    constants.newpath = str(tmp_path)
    param_ranges = {
        'z_list': np.array([6, 7, 8]),
        'm_min_list': np.array([8.5, 9.0, 9.5]),
        'target_xh_list': np.array([0.3, 0.5, 0.7]),
        'alpha_esc_list': np.array([-1.0, -0.5, 0.0]),
        'alpha_star_list': np.array([0.3, 0.5, 0.7]),
        'f_star_list': np.array([-1.5, -1.0, -0.5]),
        'tq_list': np.array([1e12, 3e13, 1e14]),
    }
    model = Models(param_ranges)
    # Check _list suffix is stripped
    assert 'z' in model.Parameters_names
    assert 'm_min' in model.Parameters_names
    assert 'z_list' not in model.Parameters_names

def test_models_rank_calculation_specific_values(tmp_path):
    """rank_calculation with Specific_values should return a single rank"""
    constants.newpath = str(tmp_path)
    param_ranges = {
        'z_list': np.array([6, 7, 8]),
        'm_min_list': np.array([8.5, 9.58, 10.5]),
        'target_xh_list': np.array([0.3, 0.5, 0.7]),
        'alpha_esc_list': np.array([-1.0, -0.5, 0.0]),
        'alpha_star_list': np.array([0.3, 0.5, 0.7]),
        'f_star_list': np.array([-1.5, -1.125, -0.5]),
        'tq_list': np.array([1e12, 3.156e13, 1e14]),
    }
    model = Models(param_ranges)
    rank = model.rank_calculation(Specific_values={'z': 7.0})
    assert isinstance(rank, list)
    assert len(rank) == 1

def test_models_rank_calculation_conflict():
    """rank_calculation should return None if P_VA and P_VO overlap"""
    param_ranges = {
        'z_list': np.array([6, 7, 8]),
        'm_min_list': np.array([8.5, 9.58, 10.5]),
        'target_xh_list': np.array([0.3, 0.5, 0.7]),
        'alpha_esc_list': np.array([-1.0, -0.5, 0.0]),
        'alpha_star_list': np.array([0.3, 0.5, 0.7]),
        'f_star_list': np.array([-1.5, -1.125, -0.5]),
        'tq_list': np.array([1e12, 3.156e13, 1e14]),
    }
    model = Models(param_ranges)
    result = model.rank_calculation(P_VA=['z'], P_VO=['z'])
    assert result is None