import numpy as np
import pytest

from orbitcal import gauss_newton
from orbitcal import FitResult

# x values used by the two-parameter and nonlinear problems
X = np.array([0, 1, 2, 3], dtype=np.float64)


# helper functions used for testing
# simple valid problem, r(x) = x - 3, used when testing invalid inputs
def simple_residual(params):
    return params - 3


def simple_jacobian(params):
    return np.ones((1, 1))


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

    @pytest.mark.parametrize(
        "tol",
        [
            0,
            -1e-8,
            np.nan,
            np.inf,
            -np.inf,
            True,
            "1e-8",
        ],
    )
    def test_invalid_tol(self, tol):
        """Check ValueError raised when tol is not positive and finite"""

        with pytest.raises(ValueError):
            gauss_newton(simple_residual, simple_jacobian, [0.0], tol=tol)

    @pytest.mark.parametrize(
        "max_iter",
        [
            0,
            -5,
            2.5,
            10.0,
            True,
            False,
            "50",
        ],
    )
    def test_invalid_max_iter(self, max_iter):
        """Check ValueError raised when max_iter is not a positive integer"""

        with pytest.raises(ValueError):
            gauss_newton(
                simple_residual, simple_jacobian, [0.0], max_iter=max_iter
            )

    @pytest.mark.parametrize(
        "initial_params",
        [
            [],
            0.0,
            [[0.0]],
            [np.nan],
            [np.inf],
        ],
    )
    def test_invalid_initial_params(self, initial_params):
        """Check ValueError raised when initial_params is not a non-empty,
        finite 1D array"""

        with pytest.raises(ValueError):
            gauss_newton(simple_residual, simple_jacobian, initial_params)

    @pytest.mark.parametrize(
        "residual_fn, jacobian_fn",
        [
            (None, simple_jacobian),
            (simple_residual, None),
            ([1.0], simple_jacobian),
            (simple_residual, np.ones((1, 1))),
        ],
    )
    def test_non_callable_functions(self, residual_fn, jacobian_fn):
        """Check ValueError raised when residual_fn or jacobian_fn is not
        callable"""

        with pytest.raises(ValueError):
            gauss_newton(residual_fn, jacobian_fn, [0.0])

    @pytest.mark.parametrize("max_iter", [1, 50, np.int64(50)])
    def test_valid_max_iter_accepted(self, max_iter):
        """Check valid integer types for max_iter, including NumPy
        integers, are accepted"""

        result = gauss_newton(
            simple_residual, simple_jacobian, [0.0], max_iter=max_iter
        )

        assert isinstance(result, FitResult)
