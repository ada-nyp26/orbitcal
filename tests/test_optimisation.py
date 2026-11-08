import numpy as np
import pytest

from orbitcal import gauss_newton
from orbitcal import FitResult

# x values used by the two-parameter and nonlinear problems
X = np.array([0, 1, 2, 3], dtype=np.float64)


class TestGaussNewton:
    """Test for gauss_newton() in optimisation.py"""

    @pytest.mark.parametrize(
        "residual_fn, jacobian_fn, initial_params, " "expected_params",
        [
            # case 1 - one parameter, r(x) = x - 3
            (
                lambda p: p - 3,
                # jacobian
                lambda p: np.array([[1.0]]),
                # initial guess
                np.array([0.0]),
                # expected parameters
                np.array([3.0]),
            ),
            # case 2 - two parameter straight line,
            # data y = 2 + 3x, r = a + b*x - y
            (
                lambda p: p[0] + p[1] * X - (2 + 3 * X),
                # jacobian
                lambda p: np.column_stack((np.ones_like(X), X)),
                # initial guess
                np.array([0.0, 0.0]),
                # expected parameters
                np.array([2.0, 3.0]),
            ),
            # case 3 - nonlinear, data y = exp(0.5x),
            # r = exp(k*x) - y
            (
                lambda p: np.exp(p[0] * X) - np.exp(0.5 * X),
                # jacobian
                lambda p: (X * np.exp(p[0] * X)).reshape(-1, 1),
                # initial guess (close to the answer)
                np.array([0.4]),
                # expected parameters
                np.array([0.5]),
            ),
        ],
    )
    def test_gauss_newton(
        self, residual_fn, jacobian_fn, initial_params, expected_params
    ):
        """Test gauss_newton() finds the known minimiser of each problem"""

        result = gauss_newton(residual_fn, jacobian_fn, initial_params)

        assert isinstance(result, FitResult)
        assert result.converged is True
        np.testing.assert_allclose(
            result.parameters, expected_params, rtol=0, atol=1e-10
        )
        assert result.cost == pytest.approx(0.0, abs=1e-20)
