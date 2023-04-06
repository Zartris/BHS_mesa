import unittest
import numpy as np
from ABM.space_continuous.utils import _differential_drive_inverse_kinematics, _differential_drive_forward_kinematics, \
    linear_to_angular_velocity


class TestDifferentialDriveKinematics(unittest.TestCase):

    def test_inverse_kinematics(self):
        wheel_radius = 0.1
        wheel_distance = 0.5
        v = 1.0
        omega = 0.5

        w_left, w_right = _differential_drive_inverse_kinematics(v, omega, wheel_radius, wheel_distance)

        self.assertIsNotNone(w_left)
        self.assertIsNotNone(w_right)

        self.assertAlmostEqual(w_left, 8.75, places=2)
        self.assertAlmostEqual(w_right, 11.25, places=2)

        # You can also add more specific test cases or assertions here to ensure the function is working as expected

    def test_forward_kinematics(self):
        wheel_radius = 0.1
        wheel_distance = 0.5
        w_right = 5.0
        w_left = 3.0

        v, omega = _differential_drive_forward_kinematics(w_right, w_left, wheel_radius, wheel_distance)

        self.assertIsNotNone(v)
        self.assertIsNotNone(omega)
        self.assertEqual(v, 0.4)
        self.assertAlmostEqual(omega, 0.3999999999999999, places=2)

    def test_linear_to_angular_velocity_conversion(self):
        wheel_radius = 0.1
        linear_velocity = 1.0

        angular_velocity = linear_to_angular_velocity(linear_velocity, wheel_radius)

        expected_angular_velocity = 10.0
        self.assertAlmostEqual(angular_velocity, expected_angular_velocity, places=5)


if __name__ == '__main__':
    unittest.main()
