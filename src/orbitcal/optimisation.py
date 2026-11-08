from dataclasses import dataclass

import numpy as np

__all__ = ["FitResult", "gauss_newton"]


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

    # copy so the caller's array is never altered
    params = np.array(initial_params, dtype=np.float64)

    residuals = np.asarray(residual_fn(params), dtype=np.float64)
    initial_cost = 0.5 * np.sum(residuals**2)

    # already a perfect fit, nothing to do
    if np.all(residuals == 0):
        return FitResult(params, 0.0, 0.0, 0, True)

    for iteration in range(1, max_iter + 1):
        jacobian = np.asarray(jacobian_fn(params), dtype=np.float64)

        # solve J * step = -r in the least-squares sense
        step = np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]

        new_params = params + step
        residuals = np.asarray(residual_fn(new_params), dtype=np.float64)

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
