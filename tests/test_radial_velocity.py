import numpy as np
import pytest

from orbitcal import radial_velocity


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
