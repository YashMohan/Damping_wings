"""
Unit tests for the fisher_matrix module.
These tests do not require 21cmFAST or simulation output files.
Full pipeline tests (differentiation, fisher_matrix function) are
covered by the examples/ scripts which use real simulation data.
"""
import pytest
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """Fixed random state for reproducible tests."""
    return np.random.default_rng(seed=42)

@pytest.fixture
def mock_transmission_profiles(rng) -> NDArray[np.float64]:
    """Synthetic transmission profiles — shape (N_sightlines, N_pixels)."""
    return rng.uniform(0.0, 1.0, size=(100, 50)).astype(np.float64)

@pytest.fixture
def mock_proximity_zone() -> NDArray[np.float64]:
    """Synthetic proximity zone radii in Mpc."""
    return np.linspace(0.5, 5.0, 10).astype(np.float64)

@pytest.fixture
def mock_derivative_matrix(rng) -> NDArray[np.float64]:
    """Synthetic derivative matrix dData/dTheta — shape (N_params, N_pixels)."""
    return rng.standard_normal(size=(4, 50)).astype(np.float64)

@pytest.fixture
def mock_sampled_distribution(rng) -> NDArray[np.float64]:
    """Synthetic sampled distribution — shape (N_sample, N_pixels)."""
    return rng.standard_normal(size=(500, 50)).astype(np.float64)


# ── add_proxy tests ───────────────────────────────────────────────────────────

class TestAddProxy:

    def test_returns_array_same_shape(self, mock_transmission_profiles, mock_proximity_zone):
        from damping_wings.fisher_matrix import add_proxy
        result = add_proxy(mock_transmission_profiles, z=7.0, R=mock_proximity_zone, seed=42)
        assert result.shape == mock_transmission_profiles.shape

    def test_does_not_modify_input(self, mock_transmission_profiles, mock_proximity_zone):
        """add_proxy must not modify the original array — uses .copy() internally."""
        from damping_wings.fisher_matrix import add_proxy
        original = mock_transmission_profiles.copy()
        add_proxy(mock_transmission_profiles, z=7.0, R=mock_proximity_zone, seed=42)
        np.testing.assert_array_equal(mock_transmission_profiles, original)

    def test_proximity_zone_pixels_modified(self, mock_transmission_profiles, mock_proximity_zone):
        """Pixels within the proximity zone should be modified."""
        from damping_wings.fisher_matrix import add_proxy
        result = add_proxy(mock_transmission_profiles, z=7.0, R=mock_proximity_zone, seed=42)
        n_proxy = len(mock_proximity_zone)
        # Proximity zone pixels should differ from original
        assert not np.allclose(
            result[:, :n_proxy],
            mock_transmission_profiles[:, :n_proxy]
        )

    def test_pixels_outside_proximity_unchanged(self, mock_transmission_profiles, mock_proximity_zone):
        """Pixels outside the proximity zone should be unchanged."""
        from damping_wings.fisher_matrix import add_proxy
        result = add_proxy(mock_transmission_profiles, z=7.0, R=mock_proximity_zone, seed=42)
        n_proxy = len(mock_proximity_zone)
        np.testing.assert_array_equal(
            result[:, n_proxy:],
            mock_transmission_profiles[:, n_proxy:]
        )

    def test_reproducible_with_same_seed(self, mock_transmission_profiles, mock_proximity_zone):
        """Same seed should produce identical results."""
        from damping_wings.fisher_matrix import add_proxy
        result1 = add_proxy(mock_transmission_profiles, z=7.0, R=mock_proximity_zone, seed=99)
        result2 = add_proxy(mock_transmission_profiles, z=7.0, R=mock_proximity_zone, seed=99)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_differ(self, mock_transmission_profiles, mock_proximity_zone):
        """Different seeds should produce different results."""
        from damping_wings.fisher_matrix import add_proxy
        result1 = add_proxy(mock_transmission_profiles, z=7.0, R=mock_proximity_zone, seed=1)
        result2 = add_proxy(mock_transmission_profiles, z=7.0, R=mock_proximity_zone, seed=2)
        assert not np.allclose(result1, result2)

    @pytest.mark.parametrize("redshift", [6.0, 6.5, 7.0])
    def test_supported_redshifts(self, mock_transmission_profiles, mock_proximity_zone, redshift):
        """add_proxy should run without error for all supported redshifts."""
        from damping_wings.fisher_matrix import add_proxy
        result = add_proxy(mock_transmission_profiles, z=redshift, R=mock_proximity_zone, seed=42)
        assert result.shape == mock_transmission_profiles.shape


# ── add_noise tests ───────────────────────────────────────────────────────────

class TestAddNoise:

    def test_returns_array_same_shape(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import add_noise
        result = add_noise(mock_transmission_profiles, SNR_A=10.0, SNR_M=10.0, seed=42)
        assert result.shape == mock_transmission_profiles.shape

    def test_output_differs_from_input(self, mock_transmission_profiles):
        """Noisy output should differ from clean input."""
        from damping_wings.fisher_matrix import add_noise
        result = add_noise(mock_transmission_profiles, SNR_A=10.0, SNR_M=10.0, seed=42)
        assert not np.allclose(result, mock_transmission_profiles)

    def test_high_snr_close_to_input(self, mock_transmission_profiles):
        """Very high SNR should produce output close to input."""
        from damping_wings.fisher_matrix import add_noise
        result = add_noise(mock_transmission_profiles, SNR_A=1e6, SNR_M=1e6, seed=42)
        np.testing.assert_allclose(result, mock_transmission_profiles, atol=1e-3)

    def test_reproducible_with_same_seed(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import add_noise
        result1 = add_noise(mock_transmission_profiles, SNR_A=10.0, SNR_M=10.0, seed=99)
        result2 = add_noise(mock_transmission_profiles, SNR_A=10.0, SNR_M=10.0, seed=99)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_differ(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import add_noise
        result1 = add_noise(mock_transmission_profiles, SNR_A=10.0, SNR_M=10.0, seed=1)
        result2 = add_noise(mock_transmission_profiles, SNR_A=10.0, SNR_M=10.0, seed=2)
        assert not np.allclose(result1, result2)

    def test_lower_snr_higher_variance(self, mock_transmission_profiles):
        """Lower SNR should produce higher variance in output."""
        from damping_wings.fisher_matrix import add_noise
        high_snr = add_noise(mock_transmission_profiles, SNR_A=100.0, SNR_M=100.0, seed=42)
        low_snr  = add_noise(mock_transmission_profiles, SNR_A=1.0,   SNR_M=1.0,   seed=42)
        assert np.var(low_snr) > np.var(high_snr)


# ── sampler tests ─────────────────────────────────────────────────────────────

class TestSampler:

    def test_returns_four_arrays(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler
        result = sampler(mock_transmission_profiles, N_data_points=10, N_sample=20, seed=42)
        assert len(result) == 4

    def test_output_shapes(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler
        n_pixels = mock_transmission_profiles.shape[1]
        mean_med, var_med, mean_sw, var_sw = sampler(
            mock_transmission_profiles, N_data_points=10, N_sample=20, seed=42
        )
        assert mean_med.shape == (n_pixels,)
        assert var_med.shape  == (n_pixels,)
        assert mean_sw.shape  == (n_pixels,)
        assert var_sw.shape   == (n_pixels,)

    def test_variance_non_negative(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler
        _, var_med, _, var_sw = sampler(
            mock_transmission_profiles, N_data_points=10, N_sample=50, seed=42
        )
        assert np.all(var_med >= 0)
        assert np.all(var_sw  >= 0)

    def test_scatter_width_non_negative(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler
        _, _, mean_sw, _ = sampler(
            mock_transmission_profiles, N_data_points=10, N_sample=50, seed=42
        )
        assert np.all(mean_sw >= 0)

    def test_reproducible_with_same_seed(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler
        r1 = sampler(mock_transmission_profiles, N_data_points=10, N_sample=20, seed=7)
        r2 = sampler(mock_transmission_profiles, N_data_points=10, N_sample=20, seed=7)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_more_samples_lower_variance(self, mock_transmission_profiles):
        """More bootstrap samples should reduce variance of the mean estimate."""
        from damping_wings.fisher_matrix import sampler
        _, var_small, _, _ = sampler(mock_transmission_profiles, N_data_points=10, N_sample=200,  seed=42)
        _, var_large, _, _ = sampler(mock_transmission_profiles, N_data_points=100, N_sample=500, seed=42)
        assert np.mean(var_large) < np.mean(var_small)


# ── sampler_dist tests ────────────────────────────────────────────────────────

class TestSamplerDist:

    def test_returns_two_arrays(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler_dist
        result = sampler_dist(mock_transmission_profiles, N_data_points=10, N_sample=20, seed=42)
        assert len(result) == 2

    def test_output_shapes(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler_dist
        N_sample = 30
        n_pixels = mock_transmission_profiles.shape[1]
        med, sw = sampler_dist(mock_transmission_profiles, N_data_points=10, N_sample=N_sample, seed=42)
        assert med.shape == (N_sample, n_pixels)
        assert sw.shape  == (N_sample, n_pixels)

    def test_scatter_width_non_negative(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler_dist
        _, sw = sampler_dist(mock_transmission_profiles, N_data_points=10, N_sample=30, seed=42)
        assert np.all(sw >= 0)

    def test_reproducible_with_same_seed(self, mock_transmission_profiles):
        from damping_wings.fisher_matrix import sampler_dist
        r1 = sampler_dist(mock_transmission_profiles, N_data_points=10, N_sample=20, seed=5)
        r2 = sampler_dist(mock_transmission_profiles, N_data_points=10, N_sample=20, seed=5)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)


# ── compute_corr_fisher_matrix tests ─────────────────────────────────────────

class TestComputeCorrFisherMatrix:

    def test_returns_two_matrices(self, mock_sampled_distribution, mock_derivative_matrix):
        from damping_wings.fisher_matrix import compute_corr_fisher_matrix
        fm, cov = compute_corr_fisher_matrix(
            mock_sampled_distribution, mock_derivative_matrix, len_theta=4
        )
        assert fm.shape  == (4, 4)
        assert cov.shape == (4, 4)

    def test_fisher_matrix_symmetric(self, mock_sampled_distribution, mock_derivative_matrix):
        from damping_wings.fisher_matrix import compute_corr_fisher_matrix
        fm, _ = compute_corr_fisher_matrix(
            mock_sampled_distribution, mock_derivative_matrix, len_theta=4
        )
        np.testing.assert_allclose(fm, fm.T, atol=1e-10)

    def test_covariance_matrix_symmetric(self, mock_sampled_distribution, mock_derivative_matrix):
        from damping_wings.fisher_matrix import compute_corr_fisher_matrix
        _, cov = compute_corr_fisher_matrix(
            mock_sampled_distribution, mock_derivative_matrix, len_theta=4
        )
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_covariance_is_inverse_of_fisher(self, mock_sampled_distribution, mock_derivative_matrix):
        """FM * FM^{-1} should equal identity."""
        from damping_wings.fisher_matrix import compute_corr_fisher_matrix
        fm, cov = compute_corr_fisher_matrix(
            mock_sampled_distribution, mock_derivative_matrix, len_theta=4
        )
        product = np.matmul(fm, cov)
        np.testing.assert_allclose(product, np.eye(4), atol=1e-8)


# ── compute_uncorr_fisher_matrix tests ───────────────────────────────────────

class TestComputeUncorrFisherMatrix:

    def test_returns_two_matrices(self, mock_derivative_matrix):
        from damping_wings.fisher_matrix import compute_uncorr_fisher_matrix
        variance = np.abs(np.random.default_rng(42).standard_normal(50)) + 0.1
        fm, cov = compute_uncorr_fisher_matrix(variance, mock_derivative_matrix, len_theta=4)
        assert fm.shape  == (4, 4)
        assert cov.shape == (4, 4)

    def test_fisher_matrix_symmetric(self, mock_derivative_matrix):
        from damping_wings.fisher_matrix import compute_uncorr_fisher_matrix
        variance = np.abs(np.random.default_rng(42).standard_normal(50)) + 0.1
        fm, _ = compute_uncorr_fisher_matrix(variance, mock_derivative_matrix, len_theta=4)
        np.testing.assert_allclose(fm, fm.T, atol=1e-10)

    def test_covariance_is_inverse_of_fisher(self, mock_derivative_matrix):
        from damping_wings.fisher_matrix import compute_uncorr_fisher_matrix
        variance = np.abs(np.random.default_rng(42).standard_normal(50)) + 0.1
        fm, cov = compute_uncorr_fisher_matrix(variance, mock_derivative_matrix, len_theta=4)
        product = np.matmul(fm, cov)
        np.testing.assert_allclose(product, np.eye(4), atol=1e-8)

    def test_higher_variance_larger_uncertainty(self, mock_derivative_matrix):
        """Higher variance in data should produce larger parameter uncertainties."""
        from damping_wings.fisher_matrix import compute_uncorr_fisher_matrix
        low_var  = np.ones(50) * 0.01
        high_var = np.ones(50) * 1.0
        _, cov_low  = compute_uncorr_fisher_matrix(low_var,  mock_derivative_matrix, len_theta=4)
        _, cov_high = compute_uncorr_fisher_matrix(high_var, mock_derivative_matrix, len_theta=4)
        assert np.trace(cov_high) > np.trace(cov_low)


# ── plotting_fisher_matrix tests ──────────────────────────────────────────────

class TestPlottingFisherMatrix:

    def test_returns_figure(self):
        from damping_wings.fisher_matrix import plotting_fisher_matrix
        sample = np.random.default_rng(42).multivariate_normal(
            mean=[0.5, 10.0, 9.5, 11.0],
            cov=np.diag([0.01, 0.1, 0.05, 0.1]),
            size=500
        )
        fig = plotting_fisher_matrix(sample)
        assert isinstance(fig, Figure)

    def test_figure_has_correct_number_of_axes(self):
        """4 parameters → 4x4 = 16 axes in corner plot."""
        from damping_wings.fisher_matrix import plotting_fisher_matrix
        sample = np.random.default_rng(42).multivariate_normal(
            mean=[0.5, 10.0, 9.5, 11.0],
            cov=np.diag([0.01, 0.1, 0.05, 0.1]),
            size=500
        )
        fig = plotting_fisher_matrix(sample)
        assert len(fig.axes) == 16

    def test_closes_cleanly(self):
        """Figure should close without error."""
        import matplotlib.pyplot as plt
        from damping_wings.fisher_matrix import plotting_fisher_matrix
        sample = np.random.default_rng(42).multivariate_normal(
            mean=[0.5, 10.0, 9.5, 11.0],
            cov=np.diag([0.01, 0.1, 0.05, 0.1]),
            size=500
        )
        fig = plotting_fisher_matrix(sample)
        plt.close(fig)


# ── differentiation guard tests ───────────────────────────────────────────────

class TestDifferentiationGuard:

    def test_raises_on_identical_ranks(self):
        """differentiation should raise ValueError if ranks point to identical parameters."""
        from damping_wings.fisher_matrix import differentiation
        import unittest.mock as mock

        # Mock a model with identical parameter values at rank 0 and rank 1
        mock_model = mock.MagicMock()
        mock_model.rank_list = {
            0: (7.0, 9.5, 0.5, -0.5, 0.5, -1.125, 3.156e13),
            1: (7.0, 9.5, 0.5, -0.5, 0.5, -1.125, 3.156e13),  # identical — no difference
        }

        with pytest.raises(ValueError, match="No parameter difference found"):
            # We use mock.patch to avoid needing real HDF5 files
            with mock.patch("h5py.File"), \
                 mock.patch("damping_wings.fisher_matrix.add_proxy", return_value=np.zeros((10, 50))), \
                 mock.patch("damping_wings.fisher_matrix.add_noise",  return_value=np.zeros((10, 50))), \
                 mock.patch("damping_wings.fisher_matrix.sampler",    return_value=(
                     np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50))):
                differentiation(
                    ranks=[0, 1],
                    N_data_points=10,
                    N_sample=20,
                    SNR_A=10.0,
                    SNR_M=10.0,
                    z=np.linspace(6.0, 8.0, 50),
                    R=np.linspace(0.5, 5.0, 10),
                    model=mock_model,
                    M_qso_base=4,
                    M_qso_order=11,
                    seed=42,
                    noise=True
                )
