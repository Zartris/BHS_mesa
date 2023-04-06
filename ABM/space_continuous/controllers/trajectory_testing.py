import math

# Robot constraints
v_max = 1.0  # m/s
w_max = 1.0  # rad/s
a_max = 1.0  # m/s^2
alpha_max = 1.0  # rad/s^2

# Waypoints (x, y, theta)
waypoints = [
    (0, 0, 0),
    (1, 1, math.pi / 4),
    (2, 1, math.pi / 2),
]


def trapezoidal_profile(distance, max_vel, max_accel):
    t_ramp = max_vel / max_accel
    d_ramp = 0.5 * max_accel * t_ramp ** 2

    if distance < 2 * d_ramp:
        t_ramp = math.sqrt(distance / max_accel)
        t_flat = 0
    else:
        t_flat = (distance - 2 * d_ramp) / max_vel

    return t_ramp, t_flat


def compute_trajectory(waypoints):
    trajectory = []

    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        distance = math.sqrt(dx ** 2 + dy ** 2)
        dtheta = end[2] - start[2]

        # Compute time profiles for linear and angular movements
        t_ramp_linear, t_flat_linear = trapezoidal_profile(distance, v_max, a_max)
        t_ramp_angular, t_flat_angular = trapezoidal_profile(abs(dtheta), w_max, alpha_max)

        # Compute total time for this segment
        t_total = max(t_ramp_linear * 2 + t_flat_linear, t_ramp_angular * 2 + t_flat_angular)

        trajectory.append({
            'start': start,
            'end': end,
            'distance': distance,
            'dtheta': dtheta,
            't_ramp_linear': t_ramp_linear,
            't_flat_linear': t_flat_linear,
            't_ramp_angular': t_ramp_angular,
            't_flat_angular': t_flat_angular,
            't_total': t_total,
        })

    return trajectory


trajectory = compute_trajectory(waypoints)
for i, segment in enumerate(trajectory):
    print(f"Segment {i}: {segment}")
