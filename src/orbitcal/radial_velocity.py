import numpy as np


__all__ = ["radial_velocity"]


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
    """

    # create copies of given arrays ensuring original arrays are not altered
    time_array = np.array(times, dtype=np.float64)
    parameters_array = np.array(parameters, dtype=np.float64)

    # check if given time array is valid, can be checked after making a copy
    # as elements are not changed
    # time array invalid when: empty, not 1D, any element not finite
    if time_array.ndim != 1 or len(time_array) == 0 or not np.all(
            np.isfinite(time_array)):
        raise ValueError("Given time array was not valid.")

    v0, K, P, phi = parameters_array

    # broadcast equation over time_array
    predicted_radial_velocity = v0 + K * np.sin(((2 * np. pi *
                                                  time_array)/P) + phi)

    return predicted_radial_velocity
