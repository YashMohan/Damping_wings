"""
Integration tests for the full damping wings pipeline.
These tests run actual 21cmFAST simulations — requires ~8GB RAM.

Run with:
    pytest tests/integration/ -v

Skip with:
    pytest tests/unit/ -v
"""
import pytest
import numpy as np
import os
import h5py
import pickle

from damping_wings import Models, Get_me_M_min
from damping_wings.config import constants, Parameters

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


def test_get_me_m_min(initial_conditions):
    """Get_me_M_min should return a positive pixel mass."""
    m_p = Get_me_M_min(initial_conditions)
    assert isinstance(m_p, float)
    assert m_p > 0
    print(f"\nPixel mass: {m_p:.3e} M_sun")


def test_models_init(tmp_output):
    """Models class should initialise correctly with minimal param ranges."""
    from damping_wings import Get_me_M_min
    m_min = 9.58  # use fixed value for test speed

    param_ranges = {
        'z_list':          np.array([7.0]),
        'm_min_list':      np.array([m_min]),
        'target_xh_list':  np.array([0.5]),
        'alpha_esc_list':  np.array([-0.5]),
        'alpha_star_list': np.array([0.5]),
        'f_star_list':     np.array([-1.125]),
        'tq_list':         np.array([Parameters['tq']]),
    }
    model = Models(param_ranges)
    assert model.DD == len(Parameters)
    assert 'z' in model.Parameters_names
    assert len(model.rank_list) > 0


def test_ionized_box_generated(initial_conditions, cache_obj, tmp_output):
    """Running modelling should produce ionized box output files."""
    m_min = 9.58

    param_ranges = {
        'z_list':          np.array([7.0]),
        'm_min_list':      np.array([m_min]),
        'target_xh_list':  np.array([0.5]),
        'alpha_esc_list':  np.array([-0.5]),
        'alpha_star_list': np.array([0.5]),
        'f_star_list':     np.array([-1.125]),
        'tq_list':         np.array([Parameters['tq']]),
    }
    model = Models(param_ranges)
    rank = model.rank_calculation(Specific_values={'z': 7.0})
    model.modelling(initial_conditions, cache_obj, rank=rank)

    # Check ionized box file was created
    expected_file = os.path.join(
        constants.newpath,
        f"Ionized_box_rank_{rank[0]}_no_halofield_"
        f"DIM_{constants.DIM}_HII_{constants.HII_DIM}_"
        f"L_{constants.L_Box}_N_{constants.N_sightlines}_"
        f"seed_{constants.seed}.p"
    )
    assert os.path.exists(expected_file), \
        f"Expected ionized box file not found: {expected_file}"

    # Check file is readable and contains valid data
    box = pickle.load(open(expected_file, 'rb'))
    assert box.shape == (constants.HII_DIM,
                         constants.HII_DIM,
                         constants.HII_DIM)
    assert np.all(box >= 0) and np.all(box <= 1), \
        "Neutral fraction must be between 0 and 1"
    print(f"\nMean neutral fraction: {box.mean():.3f}")


def test_neutral_fraction_close_to_target(initial_conditions, cache_obj, tmp_output):
    """Generated box neutral fraction should be close to target_xh."""
    target_xh = 0.5
    m_min = 9.58

    param_ranges = {
        'z_list':          np.array([7.0]),
        'm_min_list':      np.array([m_min]),
        'target_xh_list':  np.array([target_xh]),
        'alpha_esc_list':  np.array([-0.5]),
        'alpha_star_list': np.array([0.5]),
        'f_star_list':     np.array([-1.125]),
        'tq_list':         np.array([Parameters['tq']]),
    }
    model = Models(param_ranges)
    rank = model.rank_calculation(Specific_values={'z': 7.0})
    model.modelling(initial_conditions, cache_obj, rank=rank)

    expected_file = os.path.join(
        constants.newpath,
        f"Ionized_box_rank_{rank[0]}_no_halofield_"
        f"DIM_{constants.DIM}_HII_{constants.HII_DIM}_"
        f"L_{constants.L_Box}_N_{constants.N_sightlines}_"
        f"seed_{constants.seed}.p"
    )
    box = pickle.load(open(expected_file, 'rb'))
    mean_xh = box.mean()

    # Allow 10% tolerance — calibration is approximate
    assert abs(mean_xh - target_xh) < 0.1, \
        f"Mean xH {mean_xh:.3f} too far from target {target_xh}"


def test_damping_wing_output_files(initial_conditions, cache_obj, tmp_output):
    """Full pipeline should produce skewer and quantile HDF5 files."""
    m_min = 9.58

    param_ranges = {
        'z_list':          np.array([7.0]),
        'm_min_list':      np.array([m_min]),
        'target_xh_list':  np.array([0.5]),
        'alpha_esc_list':  np.array([-0.5]),
        'alpha_star_list': np.array([0.5]),
        'f_star_list':     np.array([-1.125]),
        'tq_list':         np.array([Parameters['tq']]),
    }
    model = Models(param_ranges)
    rank = model.rank_calculation(Specific_values={'z': 7.0})
    model.modelling(initial_conditions, cache_obj, rank=rank)

    # Check at least one skewer file exists
    skewer_files = [
        f for f in os.listdir(constants.newpath)
        if f.startswith('skewers_HM_') and f.endswith('.h5')
    ]
    assert len(skewer_files) > 0, "No skewer HDF5 files found"

    # Check skewer file contains required datasets
    skewer_path = os.path.join(constants.newpath, skewer_files[0])
    with h5py.File(skewer_path, 'r') as f:
        assert 'lambda' in f
        assert 'e_tau_avg' in f
        assert 'tau_avg' in f
        e_tau_avg = f['e_tau_avg'][:]
        assert np.all(e_tau_avg >= 0)
        assert np.all(e_tau_avg <= 1), \
            "Transmission must be between 0 and 1"
        print(f"\nMean transmission: {e_tau_avg.mean():.3f}")