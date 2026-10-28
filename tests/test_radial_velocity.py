import numpy as np
import pytest

from orbitcal import radial_velocity
from orbitcal import radial_velocity_jacobian


class TestRadialVelocity:
    """Test for radial_velocity() in radial_velocity.py"""

    @pytest.mark.parametrize("times, parameters, expected_velocities",
                             [
                                 # time array - case 1
                                 (np.array([0, 1, 2, 3, 4]),
                                  # parameters
                                  np.array([10, 20, 4, 0]),
                                  # expected velocities array
                                  np.array([10, 30, 10, -10, 10])),

                                 # time array - case 2
                                 (np.array([0, 1, 2, -1]),
                                  # parameters
                                  np.array([10, 20, 4, np.pi/2]),
                                  # expected velocities array
                                  np.array([30, 10, -10, 10])),
                             ])
    def test_radial_velocity(self, times, parameters, expected_velocities):
        """Test equation in radial_velocity() gives correct outputs"""

        predicted_radial_velocity = radial_velocity(times, parameters)

        assert np.allclose(predicted_radial_velocity, expected_velocities,
                           atol=1e-12)

    @pytest.mark.parametrize("times", [
        (np.array([[1, 2], [2, 3]])),
        [],
        np.array([1, 2, np.inf]),
    ])
    def test_invalid_times(self, times):
        """Check if value error raised when an invalid time array is given"""

        # random parameters since their values are meaningless here
        parameters = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError):
            radial_velocity(times, parameters)
        with pytest.raises(ValueError):
            radial_velocity_jacobian(times, parameters)

    @pytest.mark.parametrize("parameters", [
        (np.array([[1, 2], [2, 3]])),
        [],
        np.array([1, 2, np.inf, 4]),
        np.array([1, 2, 3, 4, 5]),
    ])
    def test_valid_times(self, parameters):
        """Check if value error raised when an invalid parameter array is
        given"""

        times = np.array([1, 2, 3, 4])

        # test both functions, radial_velocity() and
        # radial_velocity_jacobian()
        with pytest.raises(ValueError):
            radial_velocity(times, parameters)
        with pytest.raises(ValueError):
            radial_velocity_jacobian(times, parameters)


class TestRadialVelocityJacobian:
    """Test for radial_velocity_jacobian() in radial_velocity.py"""

    @pytest.mark.parametrize("times, parameters, expected_jacobian",
                             [
                                 # case 1 - zero phase
                                 (np.array([0, 1, 2]),
                                  # parameters
                                  np.array([10, 20, 4, 0]),
                                  # expected Jacobian
                                  np.array([
                                      [1, 0, 0, 20],
                                      [1, 1, 0, 0],
                                      [1, 0, 5*np.pi, -20]
                                  ])),

                                 # case 2 - nonzero phase and negative time
                                 (np.array([0, 1, -1, 2]),
                                  # parameters
                                  np.array([10, 20, 4, np.pi/2]),
                                  # expected Jacobian
                                  np.array([
                                      [1, 1, 0, 0],
                                      [1, 0, 5*np.pi/2, -20],
                                      [1, 0, 5*np.pi/2, 20],
                                      [1, -1, 0, 0]
                                  ])),
                             ])
    def test_radial_velocity_jacobian(
        self, times, parameters, expected_jacobian
    ):
        """Test Jacobian gives correct analytical derivatives"""

        result = radial_velocity_jacobian(times, parameters)

        np.testing.assert_allclose(
            result, expected_jacobian, rtol=0, atol=1e-12
        )

    def test_P_equal_zero(self):
        """Test ValueError raised when given P parameter equals 0"""

        parameters = np.array([1, 2, 0, 3])
        times = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError):
            radial_velocity_jacobian(times, parameters)
