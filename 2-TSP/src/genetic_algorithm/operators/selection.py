from __future__ import annotations

import random

from src.genetic_algorithm.individual import Individual
from src.genetic_algorithm.population import Population


class TournamentSelection:
    """
    Selects a parent by running a tournament between random individuals.

    A number of individuals equal to the tournament size are drawn at random
    from the population, and the fittest among them is selected.
    """

    def __init__(self, tournament_size: int, rng: random.Random) -> None:
        """
        Initialize the tournament selection operator.

        Args:
            tournament_size (int): The number of competitors per tournament.
            rng (random.Random): The seeded random number generator.
        """
        self._tournament_size = tournament_size
        self._rng = rng

    def select(self, population: Population) -> Individual:
        """
        Select a single parent from the population.

        Args:
            population (Population): The population to select from.
        Returns:
            Individual: The fittest individual among the sampled competitors.
        """
        competitors: list[Individual] = self._rng.sample(
            population.individuals,
            k=min(self._tournament_size, len(population)),
        )

        return min(competitors, key=lambda individual: individual.distance)
