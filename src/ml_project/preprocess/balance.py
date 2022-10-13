import numpy as np
from typing import Tuple


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    y_str: np.ndarray,
    genres: np.ndarray,
    genre_counts: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create balanced dataset by sampling the number of points in
    second largest class from the largest class

    Args:
        X (np.ndarray): _description_
        y (np.ndarray): _description_s
        y_str (np.ndarray): _description_
        genres (np.ndarray): _description_
        genre_counts (np.ndarray): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
    """

    # Find largest class
    max_genre_index = np.argmax(genre_counts)
    max_genre = genres[max_genre_index]

    # Generate random indices of Rock genre label to remove
    np.random.seed(seed)
    rock_indices = np.argwhere(y_str == max_genre).flatten()
    remove_n_rock_samples = (
        rock_indices.shape[0]
        - np.argwhere(y_str == genres[np.argsort(genre_counts)[-2]]).shape[0]
    )
    remove_rock_indices = np.random.choice(
        rock_indices, size=remove_n_rock_samples, replace=False
    )

    # New balanced dataset
    X_bal = np.delete(X, remove_rock_indices, axis=0)
    y_str_bal = np.delete(y_str, remove_rock_indices, axis=0)
    y_bal = np.delete(y, remove_rock_indices, axis=0)

    return X_bal, y_bal, y_str_bal
