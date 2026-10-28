import numpy as np


__all__ = ["radial_velocity", "validate_array", "radial_velocity_jacobian"]


def validate_array(array, array_name, expected_shape=None):
    if (
        array.ndim != 1
        or len(array) == 0
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{array_name} must not be empty, be 1D and all "
                         f"elements must be finite.")

    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"{array_name} shape must be {expected_shape}.")


def radial_velocity(times, parameters):
    """Estimate radial velocities using sinusoidal radial velocity model

    Parameters
    ----------
    time_array : array-like, float
        Array of relative time from starting time, negative times are allowed
        shape(N,) where N is number of time points.

    parameters: array-like
        Array of parameters [v0, K, P, phi]
        v0: systemic velocity (m/s)
        K: velocity amplitude (m/s
        P: orbital period (exoplanet) (days)
        phi: phase at t = 0 (radians)
        shape(4,)

    Returns
    -------
    predicted_radial_velocity : array-like, float
        Array of radial velocities predicted at each time point given
        shape(N,) where N is number of time points.

    Raises
    ------
    ValueError
        If given time array is invalid.
        Invalid if: empty, not 1D, any element is not finite.
        If given parameters are invalid.
        Invalid if: empty, not 1D, any element is not finite, shape not (4, ).
    """

    # create copies of given arrays ensuring original arrays are not altered
    time_array = np.array(times, dtype=np.float64)
    parameters_array = np.array(parameters, dtype=np.float64)

    # check if given array is valid, can be checked after making a copy
    # as elements are not changed
    validate_array(time_array, "time_array")
    validate_array(parameters_array, "parameters_array", expected_shape=(4,))

    v0, K, P, phi = parameters_array

    # broadcast equation over time_array
    predicted_radial_velocity = v0 + K * np.sin(((2 * np. pi *
                                                  time_array)/P) + phi)

    return predicted_radial_velocity


def radial_velocity_jacobian(times, parameters):
    """Calculate the jacobian of the radial velocity model with respect to
    its four parameters

    Parameters
    ----------
    times : array-like, float
        Array of relative observation times in days, shape (N,).

    parameters : array-like, float
        Array of parameters [v0, K, P, phi], shape (4,).
        v0: systemic velocity (m/s)
        K: radial velocity amplitude (m/s)
        P: orbital period (days)
        phi: phase at t = 0 (radians)

    Returns
    -------
        jacobian : np.ndarray, float
        Jacobian matrix of shape (N, 4), where each row corresponds
        to an observation time and each column contains the partial
        derivative with respect to [v0, K, P, phi], respectively.

    Raises
    ------
    ValueError
        If given time array is invalid.
        Invalid if: empty, not 1D, any element is not finite.
        If given parameters are invalid.
        Invalid if: empty, not 1D, any element is not finite, shape not (4, ).
    """

    # create copies of given arrays ensuring original arrays are not altered
    time_array = np.array(times, dtype=np.float64)
    parameters_array = np.array(parameters, dtype=np.float64)

    # check if given array is valid, can be checked after making a copy
    # as elements are not changed
    validate_array(time_array, "time_array")
    validate_array(parameters_array, "parameters_array", expected_shape=(4,))

    v0, K, P, phi = parameters

    # a is the part inside the sin(), separated for ease since it doesn't
    # change during partial differentiation
    a = (2 * np.pi * times / P) + phi

    # partial derivatives with respect to each parameter
    partial_gamma = np.ones_like(times, dtype=np.float64)
    partial_K = np.sin(a)
    partial_P = -(2 * np.pi * times * K / P ** 2) * np.cos(a)
    partial_phi = K * np.cos(a)

    # stack each calculated jacobian, each row representing a new relative time
    jacobian = np.column_stack((partial_gamma, partial_K, partial_P,
                                partial_phi))

    return jacobian
