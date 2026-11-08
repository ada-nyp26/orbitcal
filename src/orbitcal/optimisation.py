from dataclasses import dataclass

import numpy as np

__all__ = ["FitResult"]


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
