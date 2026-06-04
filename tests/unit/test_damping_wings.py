# tests/test_damping_wings.py
import numpy as np
from damping_wings.damping_wings import I, optical_depth
from damping_wings.config.constants import n_pixels, dl

def test_I_function_at_small_x():
    """I(x) integration function should be finite for small positive x"""
    result = I(0.01)
    assert np.isfinite(result)

def test_I_function_positive():
    """I(x) should be well-defined for x in (0,1)"""
    for x in [0.01, 0.1, 0.5, 0.9, 0.99]:
        result = I(x)
        assert np.isfinite(result)

def test_optical_depth_shape():
    """optical_depth should return array of same length as z input"""
    z = np.linspace(6.0, 8.0, n_pixels)
    xh = np.ones(n_pixels) * 0.5
    den = np.ones(n_pixels) * 0.1
    tau = optical_depth(den, xh, z, 7.0)
    assert tau.shape == z.shape

def test_optical_depth_non_negative():
    """optical_depth values should be non-negative"""
    z = np.linspace(6.0, 8.0, n_pixels)
    xh = np.ones(n_pixels) * 0.5
    den = np.ones(n_pixels) * 0.1
    tau = optical_depth(den, xh, z, 7.0)
    assert np.all(tau >= 0)