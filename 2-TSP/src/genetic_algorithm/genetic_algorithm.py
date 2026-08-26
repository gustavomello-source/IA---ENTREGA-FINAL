from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path

from src.genetic_algorithm.individual import Individual
from src.genetic_algorithm.operators.crossover import OrderCrossover
from src.genetic_algorithm.operators.mutation import SwapMutation
from src.genetic_algorithm.operators.selection import TournamentSelection
from src.genetic_algorithm.population import Population
from src.tsp.tsp_instance import TSPInstance


class GeneticAlgorithm:
    """
    Genetic algorithm for the Travelling Salesman Problem.

    The algorithm evolves a population of candidate tours across a number of
    generations, using tournament selection, order crossover and swap
    mutation, while preserving the fittest individuals through elitism.

    Every generation is logged to a timestamped file inside the results
    directory.
    """

    def __init__(
        self,
        instance: TSPInstance,
        population_size: int,
        generations: int,
        elitism_size: int,
        selection: TournamentSelection,
        crossover: OrderCrossover,
        mutation: SwapMutation,
        rng: random.Random,
        results_dir: str | Path = "results",
    ) -> None:
        """
        Initialize the genetic algorithm with injected dependencies.

        Args:
            instance (TSPInstance): The TSP instance to solve.
            population_size (int): The number of individuals per generation.
            generations (int): The number of generations to evolve.
            elitism_size (int): The number of fittest individuals kept intact.
            selection (TournamentSelection): The parent selection operator.
            crossover (OrderCrossover): The crossover operator.
            mutation (SwapMutation): The mutation operator.
            rng (random.Random): The seeded random number generator.
            results_dir (str | Path): The directory where logs are written.
        """
        self._instance = instance
        self._population_size = population_size
        self._generations = generations
        self._elitism_size = elitism_size
        self._selection = selection
        self._crossover = crossover
        self._mutation = mutation
        self._rng = rng
        self._results_dir = Path(results_dir)

        self._logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        """
        Configure a dedicated logger that writes to a timestamped file.

        The results directory is created if it does not already exist.

        Returns:
            logging.Logger: The configured logger instance.
        """
        self._results_dir.mkdir(parents=True, exist_ok=True)

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path: Path = self._results_dir / f"{self._instance.name}_{timestamp}.log"

        logger: logging.Logger = logging.getLogger(
            f"ga.{self._instance.name}.{timestamp}"
        )
        logger.setLevel(logging.INFO)
        logger.propagate = False

        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        logger.addHandler(handler)

        return logger

    def run(self) -> Individual:
        """
        Execute the evolutionary loop and return the best solution found.

        Returns:
            Individual: The fittest individual discovered across all generations.
        """
        population: Population = Population.random(
            size=self._population_size,
            instance=self._instance,
            rng=self._rng,
        )

        best: Individual = population.best()

        self._logger.info(
            "Started GA for instance '%s' (dimension %d) with population %d over %d generations.",
            self._instance.name,
            self._instance.dimension,
            self._population_size,
            self._generations,
        )

        for generation in range(1, self._generations + 1):
            population = self._evolve(population)

            current_best: Individual = population.best()

            if current_best.is_fitter_than(best):
                best = current_best

            self._logger.info(
                "Generation %d/%d - best distance: %d - overall best: %d",
                generation,
                self._generations,
                current_best.distance,
                best.distance,
            )

        self._logger.info("Finished. Best distance found: %d", best.distance)

        return best

    def _evolve(self, population: Population) -> Population:
        """
        Produce the next generation from the current population.

        Args:
            population (Population): The current generation.
        Returns:
            Population: The next generation.
        """
        next_individuals: list[Individual] = []

        # Elitism: carry over the fittest individuals unchanged.
        elites: list[Individual] = population.sorted_by_fitness()[: self._elitism_size]
        next_individuals.extend(elites)

        while len(next_individuals) < self._population_size:
            parent1: Individual = self._selection.select(population)
            parent2: Individual = self._selection.select(population)

            child1_tour, child2_tour = self._crossover.crossover(
                parent1.tour, parent2.tour
            )

            child1_tour = self._mutation.mutate(child1_tour)
            child2_tour = self._mutation.mutate(child2_tour)

            next_individuals.append(
                Individual.from_tour(tour=child1_tour, instance=self._instance)
            )

            if len(next_individuals) < self._population_size:
                next_individuals.append(
                    Individual.from_tour(tour=child2_tour, instance=self._instance)
                )

        return Population(individuals=next_individuals)
