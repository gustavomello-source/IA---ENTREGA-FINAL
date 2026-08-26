from __future__ import annotations

from dataclasses import dataclass, field

from src.tsp.tsp_instance import TSPInstance


@dataclass(slots=True)
class Individual:
    """
    Represents a candidate solution (a tour) in the genetic algorithm.

    A lower distance means a fitter individual, since the TSP is a
    minimization problem.

    Attributes:
        tour (list[int]): A permutation of city indices representing the tour.
        distance (int): The total length of the tour.
    """

    tour: list[int]
    distance: int = field(default=0)

    @classmethod
    def from_tour(cls, tour: list[int], instance: TSPInstance) -> Individual:
        """
        Create an Individual from a tour, computing its distance.

        Args:
            tour (list[int]): A permutation of city indices.
            instance (TSPInstance): The TSP instance used to evaluate the tour.
        Returns:
            Individual: The evaluated individual.
        """
        distance: int = instance.calculate_tour_distance(tour=tour)

        return cls(tour=tour, distance=distance)

    def is_fitter_than(self, other: Individual) -> bool:
        """
        Check whether this individual is fitter than another one.

        Args:
            other (Individual): The individual to compare against.
        Returns:
            bool: True if this individual has a shorter tour, False otherwise.
        """
        return self.distance < other.distance
