import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import Optional

def apply_kalman_filter_2d_cv(path: np.ndarray) -> Optional[np.ndarray]:
    """
    Applies a Kalman filter with a Constant Velocity model to smooth a 2D path.

    Args:
        path: A NumPy array of shape (N, 2) representing (x, y) coordinates.

    Returns:
        A NumPy array of shape (N, 2) with the smoothed coordinates, or None if input is invalid.
    """
    if path is None or path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        print("Warning: Invalid path input for Kalman filter.")
        return None

    # Initialize Kalman Filter (Constant Velocity Model for 2D)
    # State: [x, y, dx, dy] (position and velocity in x and y)
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([path[0, 0], path[0, 1], 0., 0.])  # Initial state [x, y, dx, dy]

    # State Transition Matrix (F) - Constant Velocity
    dt = 0.1 # Assume a time step (adjust if you have timestamps)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # Measurement Function (H) - We only measure position (x, y)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # Measurement Noise Covariance (R) - Tune this based on observed noise
    # Higher values trust measurements less. Start with identity or small values.
    kf.R = np.eye(2) * 0.5 # Example value, needs tuning

    # Process Noise Covariance (Q) - Tune this based on expected motion changes
    # Represents uncertainty in the model (how much velocity can change)
    q_var = 0.05 # Example value, needs tuning
    kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=q_var)


    # Initial State Covariance (P) - Initial uncertainty
    kf.P = np.eye(4) * 10.0

    # Run the filter
    measurements = path # Our (x, y) coordinates are the measurements
    (mu, cov, _, _) = kf.batch_filter(measurements)

    # mu contains the filtered states [x, y, dx, dy] for each time step
    # We only want the smoothed positions (x, y)
    smoothed_path = mu[:, :2]

    return smoothed_path 