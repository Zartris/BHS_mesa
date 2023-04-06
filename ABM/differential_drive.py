import numpy as np
import numba as nb
from scipy.optimize import minimize


@nb.jit(nopython=True)
def wheel_velocities(v, omega, wheel_radius, wheel_distance):
    """
    Calculate the wheel velocities for a differential drive robot.

    Args:
    v: Linear velocity of the robot (m/s)
    omega: Angular velocity of the robot (rad/s)
    wheel_radius: Radius of the wheels (m)
    wheel_distance: Distance between the wheels (m)

    Returns:
    v_right: Right wheel velocity (m/s)
    v_left: Left wheel velocity (m/s)
    """
    v_right = v + (omega * wheel_distance) / 2
    v_left = v - (omega * wheel_distance) / 2

    return v_right, v_left


@nb.jit(nopython=True)
def robot_kinematics(w_right, w_left, wheel_radius, wheel_distance):
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


@nb.jit(nopython=True)
def simulate_robot(x, y, theta, w_right, w_left, wheel_radius, wheel_distance, dt, steps):
    """
    Simulate the motion of a differential drive robot.

    Args:
    x: Initial x position of the robot (m)
    y: Initial y position of the robot (m)
    theta: Initial orientation of the robot (rad)
    w_right: Right wheel angular velocity (rad/s)
    w_left: Left wheel angular velocity (rad/s)
    wheel_radius: Radius of the wheels (m)
    wheel_distance: Distance between the wheels (m)
    dt: Time step for simulation (s)
    steps: Number of simulation steps

    Returns:
    trajectory: A list of (x, y, theta) tuples representing the robot's trajectory
    """
    trajectory = [(x, y, theta)]

    for _ in range(steps):
        # Calculate the robot's linear and angular velocities
        v, omega = robot_kinematics(w_right, w_left, wheel_radius, wheel_distance)

        # Update position and orientation using Euler's method
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        trajectory.append((x, y, theta))

    return trajectory


def simulate_motion(x, y, theta, w_right, w_left, wheel_radius, wheel_distance, dt, steps):
    """
    Simulate the motion of a differential drive robot.

    Args:
    x: Initial x position of the robot (m)
    y: Initial y position of the robot (m)
    theta: Initial orientation of the robot (rad)
    w_right: Right wheel angular velocity (rad/s)
    w_left: Left wheel angular velocity (rad/s)
    wheel_radius: Radius of the wheels (m)
    wheel_distance: Distance between the wheels (m)
    dt: Time step for simulation (s)
    steps: Number of simulation steps

    Returns:
    trajectory: A list of (x, y, theta) tuples representing the robot's trajectory
    """
    trajectory = [(x, y, theta)]

    for _ in range(steps):
        # Convert wheel angular velocities to linear velocities
        v_right = w_right * wheel_radius
        v_left = w_left * wheel_radius

        # Compute the robot's linear and angular velocities
        v = (v_right + v_left) / 2
        omega = (v_right - v_left) / wheel_distance

        # Update position and orientation using Euler's method
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        trajectory.append((x, y, theta))

    return trajectory


# Define robot dynamics
def robot_dynamics(q, qdot, u):
    """
        Computes the joint acceleration of a differential drive robot.

        Args:
            q (numpy array): The current state of the robot, represented as a numpy array of length 3, with the first element
                representing the x-coordinate of the robot's position, the second element representing the y-coordinate of the
                robot's position, and the third element representing the robot's orientation in radians.
            qdot (numpy array): The current joint velocities of the robot, represented as a numpy array of length 2, with the
                first element representing the joint velocity of the left wheel and the second element representing the joint
                velocity of the right wheel, both measured in radians per second.
            u (numpy array): The control input to the robot, represented as a numpy array of length 2, with the first element
                representing the desired joint velocity of the left wheel and the second element representing the desired joint
                velocity of the right wheel, both measured in radians per second.

        Returns:
            numpy array: The joint acceleration of the robot, represented as a numpy array of length 2, with the first element
                representing the joint acceleration of the left wheel and the second element representing the joint acceleration
                of the right wheel, both measured in radians per second squared.
    """
    # Compute joint acceleration
    v = R / 2 * (qdot[0] + qdot[1])
    w = R / L * (qdot[1] - qdot[0])
    A = np.array([[np.cos(q[2]), -v * np.sin(q[2])], [np.sin(q[2]), v * np.cos(q[2])], [0, 1]])
    å = np.dot(A, np.dot(B, qdot))
    ø = np.dot(A.T, np.dot(K, q - q_des))
    qddot = np.linalg.inv(M).dot(u - å - ø - G)
    return qddot


# Define cost function for MPC
def cost_function(g, x, u_seq):
    """
        Computes the cost of the predicted trajectory over the prediction horizon.

        Args:
            x (numpy array): The current state of the robot, represented as a numpy array of length 5, with the first three elements
                representing the x, y, and theta coordinates of the robot's position, and the last two elements representing the
                linear velocity in the x and y directions of the robot, and the angular velocity of the robot in radians per second.
            u_seq (numpy array): The sequence of predicted control inputs for the robot, represented as a numpy array of shape
                (N, 2), where N is the prediction horizon. Each row of the array represents a predicted control input, with the first element
                representing the desired joint velocity of the left wheel and the second element representing the desired joint velocity of
                the right wheel, both measured in radians per second.

        Returns:
            float: The cost of the predicted trajectory over the prediction horizon.
    """
    # Compute cost
    q, qdot = x[:3], x[3:]
    # Compute cost
    cost = 0
    for k in range(N):
        q_k = q + dt * k * robot_kinematics(q, qdot)
        qdot_k = qdot + dt * k * robot_dynamics(q, qdot, u_seq[k])
        cost += np.dot(np.transpose(q_k - q_des), np.dot(Q, q_k - q_des))
        cost += np.dot(np.transpose(qdot_k - qdot_des), np.dot(R, qdot_k - qdot_des))
        cost += np.dot(np.transpose(u_seq[k]), np.dot(Q_u, u_seq[k]))
    return cost


# Define MPC controller
def mpc_controller(q, qdot):
    """
       Computes the control input to a differential drive robot using model predictive control.

       Args:
           q (numpy array): The current state of the robot, represented as a numpy array of length 3, with the first element
               representing the x-coordinate of the robot's position, the second element representing the y-coordinate of the
               robot's position, and the third element representing the robot's orientation in radians.
           qdot (numpy array): The current joint velocities of the robot, represented as a numpy array of length 2, with the
               first element representing the joint velocity of the left wheel and the second element representing the joint
               velocity of the right wheel, both measured in radians per second.

       Returns:
           numpy array: The control input to the robot, represented as a numpy array of length 2, with the first element
               representing the desired joint velocity of the left wheel and the second element representing the desired joint
               velocity of the right wheel, both measured in radians per second.
       """

    # Compute initial state and input sequences
    x_seq = np.zeros((N + 1, 5))
    x_seq[0] = np.concatenate((q, qdot))
    u_seq = np.zeros((N, 2))

    # Solve optimization problem
    for k in range(N):
        x_k = x_seq[k]
        u_k_bounds = [(u_min[i], u_max[i]) for i in range(2)]
        result = minimize(cost_function, x_k[3:], args=(x_k, u_seq), bounds=u_k_bounds)
        u_seq[k] = result.x

        # Propagate state forward
        xdot_k = robot_dynamics(x_k[:3], x_k[3:], u_seq[k])
        x_seq[k + 1] = x_seq[k] + dt * xdot_k

    # Compute control input
    u = u_seq[0]

    # Return control input
    return u


def simulate_robot(q0, qdot0, controller, num_steps, dt, set_joint_positions):
    """
    Simulates a differential drive robot using a given controller.

    Args:
        q0 (numpy array): The initial state of the robot, represented as a numpy array of length 3, with the first element
            representing the initial x-coordinate of the robot's position, the second element representing the initial
            y-coordinate of the robot's position, and the third element representing the initial orientation of the robot in
            radians.
        qdot0 (numpy array): The initial joint velocities of the robot, represented as a numpy array of length 2, with the
            first element representing the initial joint velocity of the left wheel and the second element representing the
            initial joint velocity of the right wheel, both measured in radians per second.
        controller (function): The control function used to compute the control input for the robot.
        num_steps (int): The total number of time steps to simulate.
        dt (float): The time step used in the simulation.
        set_joint_positions (function): A function that sets the joint positions of the robot in the simulation.

    Returns:
        numpy array: An array of shape (num_steps+1, 3), where each row represents the state of the robot at a given time step,
            with the first element representing the x-coordinate of the robot's position, the second element representing the
            y-coordinate of the robot's position, and the third element representing the orientation of the robot in radians.
    """
    # Initialize the state of the robot
    q = q0
    qdot = qdot0
    # Create an array to store the state of the robot at each time step
    q_history = np.zeros((num_steps + 1, 3))
    q_history[0] = q
    # Simulate the robot for the specified number of time steps
    for i in range(num_steps):
        # Compute the control input using the given controller
        u = controller(q, qdot)
        # Compute the joint acceleration of the robot
        qddot = robot_dynamics(q, qdot, u)
        # Update the joint velocities of the robot
        qdot += qddot * dt
        # Update the state of the robot
        q += robot_kinematics(q, qdot) * dt
        # Set the joint positions of the robot in the simulation
        # set_joint_positions(q[0], q[1], q[2])
        # Record the state of the robot at the current time step
        q_history[i + 1] = q
    return q_history


def set_joint_positions(x, y, theta, v_l, v_r):
    """
    Sets the joint positions of a differential drive robot.

    Args:
        x (float): The x-coordinate of the robot's position.
        y (float): The y-coordinate of the robot's position.
        theta (float): The orientation of the robot in radians.
        v_l (float): The velocity of the left wheel.
        v_r (float): The velocity of the right wheel.
    Returns:
        Tuple of the updated position and orientation of the robot.
    """

    # Compute the joint angles for the left and right wheels
    dtheta_l = v_l / R
    dtheta_r = v_r / R
    left_wheel_angle = -theta + dtheta_l
    right_wheel_angle = theta + dtheta_r
    # Set the joint positions of the left and right wheels
    q[left_wheel_indices] = left_wheel_angle
    q[right_wheel_indices] = right_wheel_angle
    # Update the position and orientation of the robot
    x += R / 2 * (dtheta_l + dtheta_r) * np.cos(theta)
    y += R / 2 * (dtheta_l + dtheta_r) * np.sin(theta)
    theta += R / L * (dtheta_r - dtheta_l)
    return (x, y, theta)


if __name__ == '__main__':
    # Define simulation parameters
    num_steps = 1000  # number of simulation steps

    # Define robot parameters
    R = 0.1  # radius of the wheels
    L = 0.3  # distance between the wheels
    M = np.diag([1, 1, 1])  # mass matrix
    B = np.diag([0.01, 0.01])  # damping matrix
    K = np.diag([10, 10])  # stiffness matrix
    G = np.array([0, 0, -9.81])  # gravity vector
    Q = np.diag([1, 1, 1])  # The weighting matrix for the state error in the cost function
    Q_u = np.diag([0.1, 0.1])  # The weighting matrix for the control effort in the cost function

    u_min = np.array([-10, -10])  # minimum control input
    u_max = np.array([10, 10])  # maximum control input
    left_wheel_indices = 0
    right_wheel_indices = 1

    # Define prediction horizon and time step
    N = 10  # prediction horizon
    dt = 0.1  # time step

    # Main simulation loop
    q = np.array([0, 0, 0])  # initial position and orientation
    qdot = np.array([0, 0])  # initial joint velocities
    q_des = np.array([100, 100, np.deg2rad(90)])  # desired position and orientation
    qdot_des = np.array([10.0, 10.0])  # desired joint velocities

    simulate_robot(q, qdot, mpc_controller, num_steps, dt, set_joint_positions)
    #
    # for t in range(num_steps):
    #     # Compute control input
    #     u = mpc_controller(q, qdot)
    #
    #     # Simulate robot dynamics
    #     qddot = robot_dynamics(q, qdot, u)
    #     qdot = qdot + dt * qddot
    #     q = q + dt * robot_kinematics(q, qdot)
    #
    #     # Update robot state
    #     set_joint_positions(q)
