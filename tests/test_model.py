import numpy as np

from worldcup_predictor.model import (
    apply_temperature,
    dixon_coles_score_matrix,
    poisson_outcome_probabilities,
)


def test_poisson_probabilities_sum_to_one():
    probabilities = poisson_outcome_probabilities(np.array([1.8, 0.9]), np.array([0.8, 1.4]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_temperature_preserves_probability_simplex():
    probabilities = np.array([[0.2, 0.3, 0.5]])
    adjusted = apply_temperature(probabilities, 1.2)
    np.testing.assert_allclose(adjusted.sum(axis=1), 1.0)
    assert adjusted.argmax(axis=1)[0] == 2


def test_dixon_coles_correction_preserves_score_probability_mass():
    matrix = dixon_coles_score_matrix(
        np.array([1.4]), np.array([1.1]), max_goals=8, rho=-0.08
    )
    np.testing.assert_allclose(matrix.sum(axis=(1, 2)), 1.0)
    assert (matrix >= 0).all()
