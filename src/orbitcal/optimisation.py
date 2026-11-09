from dataclasses import dataclass
from .radial_velocity import validate_array

import numpy as np

__all__ = ["FitResult", "gauss_newton"]


# validation helper functions to validate inputs
def validate_inputs(residual_fn, jacobian_fn, tol, max_iter):
    """Check gauss_newton() inputs are valid, raising ValueError if not"""

    if not callable(residual_fn) or not callable(jacobian_fn):
        raise ValueError("residual_fn and jacobian_fn must be callable.")

    # bool checked first as True and False count as integers
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float, np.integer, np.floating))
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError("tol must be a positive finite number.")

    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer")


def validate_residuals(residuals, n_params):
    """Check the first residuals are a finite 1D array with at least one
    value per parameter, raising ValueError if not"""

    if (
        residuals.ndim != 1
        or len(residuals) < n_params
        or not np.all(np.isfinite(residuals))
    ):
        raise ValueError(
            f"First residuals must be a finite 1D array with at least "
            f"{n_params} values (one per parameter)."
        )


def validate_jacobian(jacobian, n_residuals, n_params, iteration):
    """Check the Jacobian is finite with shape (N, M)
    Raises ValueError on first iteration (bad user input),
    but raises RuntimeError on later iterations (optimser has failed)
    """

    if jacobian.shape != (n_residuals, n_params) or not np.all(
        np.isfinite(jacobian)
    ):
        error_type = ValueError if iteration == 1 else RuntimeError
        raise error_type(
            f"jacobian at iteration {iteration} must be finite with shape"
            f"({n_residuals}, {n_params}), got shape {jacobian.shape}"
        )


@dataclass
class FitResult:
    """Result of a Gauss–Newton least-squares fit

    Attributes
    ----------
    parameters : np.ndarray, float
        Final fitted parameters, shape (M,).

    initial_cost : float
        Half sum of squared residuals at the initial guess,
        0.5 * sum(r(initial_params)**2).

    cost : float
        Half sum of squared residuals at the final parameters,
        0.5 * sum(r(parameters)**2).

    iterations : int
        Number of parameter updates performed.

    converged : bool
        True if the step-size convergence criterion was met,
        False if max_iter was reached without converging.
    """

    parameters: np.ndarray
    initial_cost: float
    cost: float
    iterations: int
    converged: bool


def gauss_newton(
    residual_fn, jacobian_fn, initial_params, tol=1e-8, max_iter=50
):
    """Minimise 0.5 * sum(r(theta)**2) using full-step Gauss–Newton

    Parameters
    ----------
    residual_fn : callable
        Takes parameters, shape (M,), returns residuals, shape (N,).

    jacobian_fn : callable
        Takes parameters, shape (M,), returns the Jacobian of the
        residuals, shape (N, M).

    initial_params : array-like, float
        Initial guess, shape (M,).

    tol : float
        Converged when norm(step) <= tol * (1 + norm(previous params)).

    max_iter : int
        Maximum number of parameter updates.

    Returns
    -------
    FitResult
        Final parameters, initial and final cost, number of iterations
        and convergence status.
    """

    validate_inputs(residual_fn, jacobian_fn, tol, max_iter)

    # copy so the caller's array is never altered
    params = np.array(initial_params, dtype=np.float64)
    validate_array(params, "initial_params")

    residuals = np.asarray(residual_fn(params), dtype=np.float64)
    validate_residuals(residuals, len(params))
    initial_cost = 0.5 * np.sum(residuals**2)

    # already a perfect fit, nothing to do
    if np.all(residuals == 0):
        return FitResult(params, 0.0, 0.0, 0, True)

    for iteration in range(1, max_iter + 1):
        jacobian = np.asarray(jacobian_fn(params), dtype=np.float64)
        validate_jacobian(jacobian, len(residuals), len(params), iteration)

        # solve J * step = -r in the least-squares sense, keeping the rank
        step, _, rank, _ = np.linalg.lstsq(jacobian, -residuals, rcond=None)

        # dependent columns mean there is no unique step
        if rank < len(params):
            raise RuntimeError(
                f"jacobian is rank-deficient at iteration {iteration} "
                f"(rank {rank} < {len(params)} parameters), so no unique "
                f"step exists."
            )

        new_params = params + step

        # the model may reject parameters the step has moved into,
        # e.g. a negative period
        try:
            new_residuals = residual_fn(new_params)
        except ValueError as error:
            raise RuntimeError(
                f"step at iteration {iteration} moved parameters to "
                f"{new_params}, where residual_fn failed: {error}"
            ) from error

        new_residuals = np.asarray(new_residuals, dtype=np.float64)
        if new_residuals.shape != residuals.shape or not np.all(
            np.isfinite(new_residuals)
        ):
            raise RuntimeError(
                f"residuals at iteration {iteration} are non-finite or "
                f"changed shape from {residuals.shape} to "
                f"{new_residuals.shape}."
            )
        residuals = new_residuals

        # compare step size against the OLD parameters, before updating
        converged = np.linalg.norm(step) <= tol * (1 + np.linalg.norm(params))
        params = new_params

        if converged:
            return FitResult(
                params,
                float(initial_cost),
                float(0.5 * np.sum(residuals**2)),
                iteration,
                True,
            )

    # ran out of iterations without meeting the convergence rule
    return FitResult(
        params,
        float(initial_cost),
        float(0.5 * np.sum(residuals**2)),
        max_iter,
        False,
    )
