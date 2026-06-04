# tests/test_utils.py

import pytest
import numpy as np
from damping_wings.utils import H, calculate_t_vir

def test_H_at_zero_redshift():
    """H(0) should equal H0 when Omega_k = 0 and Omega_m + Omega_lambda = 1"""
    from damping_wings.config.constants import H0
    result = H(0.0)
    assert abs(result - H0) < 1.0  # within 1 m/s/Mpc

def test_H_increases_with_redshift():
    """Hubble parameter should increase with redshift"""
    assert H(5.0) > H(3.0) > H(1.0) > H(0.0)

def test_H_positive():
    """Hubble parameter must always be positive"""
    for z in [0, 1, 5, 7, 10]:
        assert H(z) > 0

def test_calculate_t_vir_positive():
    """Virial temperature must be positive for physical inputs"""
    result = calculate_t_vir(z=7.0, xh=0.5, m=9.58)
    assert result > 0

def test_calculate_t_vir_increases_with_mass():
    """More massive halos should have higher virial temperature"""
    t_low = calculate_t_vir(z=7.0, xh=0.5, m=8.0)
    t_high = calculate_t_vir(z=7.0, xh=0.5, m=11.0)
    assert t_high > t_low