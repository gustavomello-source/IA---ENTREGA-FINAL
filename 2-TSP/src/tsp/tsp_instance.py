from dataclasses import dataclass, field

import numpy as np

from src.tsp.city import City
from src.utils.distances import (
    calculate_att_distance_matrix,
    calculate_euc_distance_matrix,
)


@dataclass(slots=True)
class TSPInstance:
    """
    Represents a complete TSP problem.

    Attributes:
        name (str): The name of the TSP instance.
        comment (str): Additional comments about the TSP instance.
        dimension (int): The number of cities in the TSP instance.
        edge_weight_type (str): The type of edge weights used in the TSP instance.
        cities (list[City]): A list of City objects representing the cities in the TSP instance.
    """

    name: str
    comment: str
    dimension: int
    edge_weight_type: str
    cities: list[City]
    distance_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """
        Post-initialization method to build the distance matrix after the TSPInstance is created.
        """
        self.distance_matrix = self._build_distance_matrix(cities=self.cities)

    def _build_distance_matrix(self, cities: list[City]) -> np.ndarray:
        """
        Args:
            cities (list[City]): A list of City objects representing the cities in the TSP instance.

        Returns:
            np.ndarray: A symmetric distance matrix representing the distances between cities.
        Raises:
            NotImplementedError: If the edge weight type is not supported.
        """
        edge_weight_type: str = self.edge_weight_type.upper()

        if edge_weight_type == "EUC_2D":
            return calculate_euc_distance_matrix(cities=cities)

        if edge_weight_type == "ATT":
            return calculate_att_distance_matrix(cities=cities)

        raise NotImplementedError(
            f"Edge weight type '{self.edge_weight_type}' is not supported."
        )

    def calculate_tour_distance(self, tour: list[int]) -> int:
        """
        Calculates the total distance of a given tour.

        Args:
            tour (list[int]): A list of city indices representing the tour.

        Returns:
            int: The total distance of the tour.
        """
        total = 0

        for i in range(len(tour)):
            origin: int = tour[i]
            destination: int = tour[(i + 1) % len(tour)]

            total += self.distance_matrix[origin, destination]

        return int(total)
