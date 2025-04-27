import numpy as np
from typing import Optional, Tuple

def preprocess_sequence(sequence: np.ndarray, sequence_length: int = 50) -> Optional[np.ndarray]:
    """
    Preprocesses a raw trajectory sequence for model input.

    Steps include:
    1. Handling sequences with fewer than 2 points.
    2. Translating the sequence to start at the origin (0,0 or 0,0,0).
    3. Normalizing the sequence scale based on standard deviation.
    4. Resampling the sequence to a fixed length using linear interpolation.

    Args:
        sequence: A NumPy array of shape (N, 2) or (N, 3) representing
                  (x, y) or (x, y, z) coordinates. N is the original number of points.
        sequence_length: The desired fixed length for the output sequence.

    Returns:
        A NumPy array of shape (sequence_length, num_features) with the
        preprocessed sequence, or None if the input sequence had less than 2 points.
        num_features will match the input sequence (2 or 3).
    """
    if sequence.shape[0] < 2:
        # Cannot process sequences with less than 2 points
        return None

    num_features = sequence.shape[1]

    # 1. Translate to origin
    translated_sequence = sequence - sequence[0]

    # 2. Normalize scale
    # Calculate std dev across all points and dimensions
    std_dev = np.std(translated_sequence)

    if np.isclose(std_dev, 0):
        # If std_dev is zero (e.g., all points are the same after translation),
        # return a sequence of zeros as normalization/resampling is meaningless.
        # Alternatively, could return the (identically zero) translated_sequence
        # repeated, but zeros might be safer for model input.
        return np.zeros((sequence_length, num_features))

    normalized_sequence = translated_sequence / std_dev

    # 3. Resample to fixed length
    original_length = normalized_sequence.shape[0]
    original_time = np.linspace(0, original_length - 1, original_length)
    new_time = np.linspace(0, original_length - 1, sequence_length)

    resampled_sequence = np.zeros((sequence_length, num_features))
    for i in range(num_features):
        resampled_sequence[:, i] = np.interp(new_time, original_time, normalized_sequence[:, i])

    return resampled_sequence 