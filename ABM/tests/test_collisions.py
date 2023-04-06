import time
import unittest
import numpy as np

from ABM.space_continuous.utils import check_collision


class TestCheckCollision(unittest.TestCase):
    def test_no_collision(self):
        positions = np.array([[0., 0.], [5., 5.], [10., 10.]])
        radii = np.array([1., 1., 1.])
        expected_collisions = np.zeros(3)
        result = check_collision(positions, radii)
        np.testing.assert_array_equal(result, expected_collisions)

    def test_all_agents_collide(self):
        positions = np.array([[0., 0.], [1., 1.], [2., 2.]])
        radii = np.array([1., 1., 1.])
        expected_collisions = np.ones(3)
        result = check_collision(positions, radii)
        np.testing.assert_array_equal(result, expected_collisions)

    def test_some_agents_collide(self):
        positions = np.array([[0., 0.], [1., 1.], [5., 5.]])
        radii = np.array([1., 1., 1.])
        expected_collisions = np.array([1, 1, 0])
        result = check_collision(positions, radii)
        np.testing.assert_array_equal(result, expected_collisions)

    def test_time(self):
        # warmup because numba is slow
        positions = np.random.rand(1000, 2)
        radii = np.ones(1000)
        check_collision(positions, radii)

        # real test
        positions = np.random.rand(1000, 2)
        radii = np.ones(1000)
        start = time.time()
        check_collision(positions, radii)
        end = time.time()
        print("time it took for 1000 robots", end - start)


if __name__ == '__main__':
    unittest.main()
