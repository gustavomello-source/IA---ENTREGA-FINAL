import math

import numpy as np

from src.tsp.city import City


def calculate_euc_distance_matrix(cities: list[City]) -> np.ndarray:
    """
    Calculate the TSPLIB EUC_2D distance matrix for a list of cities.

    Args:
        cities: List of City objects.

    Returns:
        A symmetric distance matrix.
    """
    n: int = len(cities)
    matrix: np.ndarray = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(i + 1, n):
            dx: float = cities[i].x - cities[j].x
            dy: float = cities[i].y - cities[j].y

            distance: int = int(np.sqrt(dx * dx + dy * dy) + 0.5)

            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix


def calculate_att_distance_matrix(cities: list[City]) -> np.ndarray:
    """
    Calculate the TSPLIB ATT (pseudo-Euclidean) distance matrix.

    The ATT rule computes r = sqrt((dx^2 + dy^2) / 10.0), rounds to the
    nearest integer t, and if t is smaller than r the distance is t + 1,
    otherwise it is t.

    Args:
        cities: List of City objects.

    Returns:
        A symmetric distance matrix.
    """
    n: int = len(cities)
    matrix: np.ndarray = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(i + 1, n):
            dx: float = cities[i].x - cities[j].x
            dy: float = cities[i].y - cities[j].y

            r: float = math.sqrt((dx * dx + dy * dy) / 10.0)
            t: int = int(r + 0.5)

            distance: int = t + 1 if t < r else t

            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix
