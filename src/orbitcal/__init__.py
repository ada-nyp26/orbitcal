"""A Python package implementing Gauss–Newton nonlinear least-squares
optimisation, applied to exoplanet orbital parameter estimation."""

from .radial_velocity import radial_velocity  # noqa
from .radial_velocity import radial_velocity_jacobian  # noqa
from .optimisation import FitResult  # noqa
from .optimisation import gauss_newton  # noqa
