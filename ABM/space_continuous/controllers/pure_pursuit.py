import time

import cv2
import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


@njit
def calculate_wheel_velocities(v, omega, wheel_radius, wheel_distance):
    """
    Calculate the wheel angular velocities for a differential drive robot.

    Args:
    v: Linear velocity of the robot (m/s)
    omega: Angular velocity of the robot (rad/s)
    wheel_radius: Radius of the wheels (m)
    wheel_distance: Distance between the wheels (m)

    Returns:
    w_left: Left wheel angular velocity (rad/s)
    w_right: Right wheel angular velocity (rad/s)
    """
    v_right = v + (omega * wheel_distance) / 2
    v_left = v - (omega * wheel_distance) / 2

    # Convert linear velocities to angular velocities
    w_right = v_right / wheel_radius
    w_left = v_left / wheel_radius

    return w_left, w_right


@njit
def calculate_wheel_velocities2(v, w, wheel_radius, wheel_distance):
    w_right = (2 * v + w * wheel_distance) / (2 * wheel_radius)
    w_left = (2 * v - w * wheel_distance) / (2 * wheel_radius)
    return w_right, w_left


@njit
def calculate_lookahead_point(path, robot_position, lookahead_distance):
    min_distance = np.inf
    lookahead_point = None

    for point in path:
        distance = np.linalg.norm(robot_position.astype(np.float64) - point.astype(np.float64))

        if distance < min_distance and distance >= lookahead_distance:
            min_distance = distance
            lookahead_point = point

    return lookahead_point


# @njit
def pure_pursuit_controller(path, robot_position, robot_orientation, lookahead_distance, wheel_radius, wheel_distance,
                            max_speed, max_acceleration, current_speed, delta_time):
    # Find the lookahead point on the path
    lookahead_point = calculate_lookahead_point(path, robot_position, lookahead_distance)

    if lookahead_point is None:
        # The robot has reached the end of the path
        return 0, 0

    # Calculate the desired heading
    delta_x = lookahead_point[0] - robot_position[0]
    delta_y = lookahead_point[1] - robot_position[1]
    desired_heading = np.arctan2(delta_y, delta_x)
    desired_heading_deg = np.rad2deg(desired_heading)
    robot_heading = np.rad2deg(robot_orientation)
    # Calculate the heading error
    heading_error = desired_heading - robot_orientation
    heading_error_deg = np.rad2deg(heading_error)

    # Calculate the desired linear and angular velocities
    v_desired = np.linalg.norm(robot_position.astype(np.float64) - lookahead_point.astype(np.float64))

    # Limit the desired speed based on max speed and acceleration constraints
    v_desired = min(v_desired, max_speed)
    v_desired = min(v_desired, current_speed + max_acceleration * delta_time)

    gain = 20
    w_desired = gain * v_desired * np.sin(heading_error) / lookahead_distance

    # Calculate the desired wheel velocities using the kinematic model
    w_left, w_right = calculate_wheel_velocities(v_desired, w_desired, wheel_radius, wheel_distance)

    return w_right, w_left


@njit
def robot_kinematics(w_left, w_right, wheel_radius, wheel_distance):
    """
    Calculate the linear and angular velocities of a differential drive robot.

    Args:
    w_right: Right wheel angular velocity (rad/s)
    w_left: Left wheel angular velocity (rad/s)
    wheel_radius: Radius of the wheels (m)
    wheel_distance: Distance between the wheels (m)

    Returns:
    v: Linear velocity of the robot (m/s)
    omega: Angular velocity of the robot (rad/s)
    """
    v_right = w_right * wheel_radius
    v_left = w_left * wheel_radius

    v = (v_right + v_left) / 2
    omega = (v_right - v_left) / wheel_distance

    return v, omega


def update_path(path, robot_position, threshold_distance=0.25):
    distances = np.linalg.norm(path - robot_position, axis=1)
    remaining_points = distances > threshold_distance
    updated_path = path[remaining_points]
    return updated_path


def generate_trajectory(path, distance_between_points=0.1):
    # Calculate the length of the path
    length = len(path)

    # Check if the path has at least 2 points
    if length < 2:
        raise ValueError("Path must have at least 2 points")

    # Calculate the distances between consecutive points in the path
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)

    # Calculate the cumulative distances along the path
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Create a cubic spline interpolator for x and y coordinates
    x_spline = CubicSpline(cumulative_distances, path[:, 0])
    y_spline = CubicSpline(cumulative_distances, path[:, 1])

    # Calculate the total distance of the path
    total_distance = cumulative_distances[-1]

    # Generate the interpolated distances based on the specified distance between points
    interpolated_distances = np.arange(0, total_distance, distance_between_points)

    # Evaluate the spline interpolator at the interpolated distances
    x_trajectory = x_spline(interpolated_distances)
    y_trajectory = y_spline(interpolated_distances)

    # Combine the x and y coordinates into the final trajectory
    trajectory = np.column_stack((x_trajectory, y_trajectory))

    return trajectory


import matplotlib.pyplot as plt


def plot_path_and_trajectory(path, trajectory):
    plt.figure(figsize=(8, 6))
    plt.plot(path[:, 0], path[:, 1], 'o-', label='Original Path', markersize=8)
    plt.plot(trajectory[:, 0], trajectory[:, 1], '.-', label='Smoothed Trajectory', markersize=4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Path and Smoothed Trajectory')
    plt.grid()
    plt.show()


def visualize_trajectory(path, trajectory, robot_position, scale=10, radius=5,
                         thickness=-1):
    # Set the dimensions for the visualization window
    width = int(max(np.max(path[:, 0]), np.max(trajectory[:, 0])) * scale) + 2 * radius
    height = int(max(np.max(path[:, 1]), np.max(trajectory[:, 1])) * scale) + 2 * radius

    # Create a black background
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Define colors
    path_color = (255, 0, 0)  # Blue
    trajectory_color = (0, 255, 0)  # Green
    robot_color = (0, 0, 255)  # Red
    # Draw path points
    for point in path:
        cv2.circle(img, tuple((point * scale).astype(int)), radius, path_color, thickness)

    # Draw trajectory points
    for point in trajectory:
        cv2.circle(img, tuple((point * scale).astype(int)), radius, trajectory_color, thickness)

    # Draw robot position
    cv2.circle(img, tuple((robot_position * scale).astype(int)), radius * 2, robot_color, thickness)

    # Display the image
    cv2.imshow("Trajectory Visualization", img)

    # Wait for a key press and close the window
    cv2.waitKey(1)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # Test the controller with a simple path
    path = np.array([[0., 1.], [1., 1.], [2., 0.], [3., 1.], [4., 0.]])
    traj = generate_trajectory(path, distance_between_points=0.1)
    robot_position = np.array([0., 0.])
    robot_orientation = 0.
    lookahead_distance = 1
    wheel_radius = 0.1
    wheel_distance = 0.5
    max_speed = 10.0
    max_acceleration = 5.0
    current_speed = 0
    dt = 1 / 240
    dt = 0.01

    w_right, w_left = pure_pursuit_controller(traj, robot_position, robot_orientation, lookahead_distance, wheel_radius,
                                              wheel_distance, max_speed, max_acceleration, current_speed, dt)

    print("Right wheel angular velocity:", w_right)
    print("Left wheel angular velocity:", w_left)

    # Test the controller with a simple path

    for t in range(10000):
        # Update the path before passing it to the MPC controller
        traj = update_path(traj, robot_position, threshold_distance=0.05)
        start = time.perf_counter()
        w_right, w_left = pure_pursuit_controller(traj, robot_position, robot_orientation, lookahead_distance,
                                                  wheel_radius,
                                                  wheel_distance, max_speed, max_acceleration, current_speed, dt)

        end = time.perf_counter()

        v, w = robot_kinematics(w_left, w_right, wheel_radius, wheel_distance)
        current_v = v

        # compute new position
        robot_orientation += w * dt
        robot_position += np.array([v * np.cos(robot_orientation), v * np.sin(robot_orientation)]) * dt
        print("Right wheel angular velocity:", w_right)
        print("Left wheel angular velocity:", w_left)
        print("Robot position:", robot_position)
        print("Robot orientation:", robot_orientation)
        print("Linear velocity:", v)
        print("Angular velocity:", w)
        print("length of path:", len(traj))
        print("Time:", end - start)
        visualize_trajectory(path, traj, robot_position, scale=1000, radius=5, thickness=-1)
