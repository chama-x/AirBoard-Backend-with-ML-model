import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple

# Define waypoints for digits 0-9 within a rough [-1, 1] box
# Ensure at least 4 points for cubic interpolation
_DIGIT_WAYPOINTS: Dict[int, List[Tuple[float, float]]] = {
    0: [(0,1), (0.5,1), (1,0.5), (1,-0.5), (0.5,-1), (-0.5,-1), (-1,-0.5), (-1,0.5), (-0.5,1), (0,1)],
    1: [(-0.25, -1), (0, -1), (0, 0), (0, 1)], # Adjusted to have 4 points
    2: [(-1, 0.8), (0, 1), (1, 0.8), (0, 0), (-1, -0.8), (0, -1), (1, -0.8)],
    3: [(-0.5, 1), (0.5, 0.8), (0, 0.5), (0.5, 0.2), (0, 0), (0.5, -0.2), (0, -0.5), (-0.5, -0.8), (0.5, -1)],
    4: [(-0.5, 1), (-0.5, 0), (1, 0), (0.5, 1), (0.5, -1)],
    5: [(1, 1), (-0.5, 1), (-0.5, 0), (0.5, 0), (1, -0.5), (0.5, -1), (-0.5, -1)],
    6: [(0.5, 1), (-0.5, 0.5), (-1, 0), (-0.5, -0.5), (0, -1), (0.5, -0.5), (0.5, 0), (-0.5, 0)],
    7: [(-1, 1), (1, 1), (0.5, 0.5), (0, 0), (-0.5, -1)], # Adjusted to have more points
    8: [(0,0), (0.5,0.5), (0,1), (-0.5,0.5), (0,0), (0.5,-0.5), (0,-1), (-0.5,-0.5), (0,0)],
    9: [(-0.5, -1), (0.5, -0.5), (1, 0), (0.5, 0.5), (0, 1), (-0.5, 0.5), (-0.5, 0), (0.5, 0)]
}

def generate_synthetic_digit(digit: int, num_points: int = 50, noise_std_dev: float = 0.03) -> np.ndarray:
    """
    Generates a synthetic 2D trajectory for a given digit using waypoints,
    interpolation, and added noise.

    Args:
        digit: The integer digit (0-9) to generate.
        num_points: The approximate number of points for the output trajectory.
        noise_std_dev: Standard deviation of the Gaussian noise added to waypoints.

    Returns:
        A NumPy array of shape (num_points, 2) representing the noisy trajectory.

    Raises:
        ValueError: If the digit is not between 0 and 9.
    """
    if digit not in _DIGIT_WAYPOINTS:
        raise ValueError("Digit must be between 0 and 9.")

    waypoints = np.array(_DIGIT_WAYPOINTS[digit])

    # Calculate cumulative distance along waypoints for interpolation parameter
    distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0) # Start at distance 0

    # Ensure distances are strictly increasing for interp1d
    # (Handles cases where waypoints might be identical)
    duplicates = np.where(np.diff(distances) <= 1e-8)[0]
    while len(duplicates) > 0:
        distances[duplicates + 1] += 1e-6 # Add tiny offset
        duplicates = np.where(np.diff(distances) <= 1e-8)[0]


    # Choose interpolation kind based on number of points
    # Need at least 4 points for cubic
    interp_kind = 'cubic' if len(distances) >= 4 else 'linear'
    if interp_kind == 'linear' and len(distances) < 2:
        # Extremely unlikely case, but handle if only one unique waypoint
        # Return the waypoint repeated num_points times with noise
        single_point = waypoints[0]
        noise = np.random.normal(scale=noise_std_dev, size=(num_points, 2))
        return np.tile(single_point, (num_points, 1)) + noise


    fx = interp1d(distances, waypoints[:, 0], kind=interp_kind)
    fy = interp1d(distances, waypoints[:, 1], kind=interp_kind)

    # Generate new points along the path
    new_distances = np.linspace(0, distances[-1], num_points)
    x_new = fx(new_distances)
    y_new = fy(new_distances)

    # Add Gaussian noise
    noise = np.random.normal(scale=noise_std_dev, size=(num_points, 2))
    trajectory = np.vstack([x_new, y_new]).T + noise

    return trajectory 