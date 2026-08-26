from __future__ import annotations

import random
from dataclasses import dataclass

from src.genetic_algorithm.individual import Individual
from src.tsp.tsp_instance import TSPInstance


@dataclass(slots=True)
class Population:
    """
    Represents a collection of individuals in a single generation.

    Attributes:
        individuals (list[Individual]): The individuals that make up the population.
    """

    individuals: list[Individual]

    def __len__(self) -> int:
        return len(self.individuals)

    def best(self) -> Individual:
        """
        Return the fittest individual (the one with the shortest tour).

        Returns:
            Individual: The individual with the smallest distance.
        """
        return min(self.individuals, key=lambda individual: individual.distance)

    def sorted_by_fitness(self) -> list[Individual]:
        """
        Return the individuals sorted from fittest to least fit.

        Returns:
            list[Individual]: Individuals ordered by ascending distance.
        """
        return sorted(self.individuals, key=lambda individual: individual.distance)

    @classmethod
    def random(
        cls,
        size: int,
        instance: TSPInstance,
        rng: random.Random,
    ) -> "Population":
        """
        Build a random initial population of valid tours.

        Args:
            size (int): The number of individuals to generate.
            instance (TSPInstance): The TSP instance used to evaluate the tours.
            rng (random.Random): The seeded random number generator.
        Returns:
            Population: A population of random permutations.
        """
        individuals: list[Individual] = []

        for _ in range(size):
            tour: list[int] = list(range(instance.dimension))
            rng.shuffle(tour)

            individuals.append(Individual.from_tour(tour=tour, instance=instance))

        return cls(individuals=individuals)
