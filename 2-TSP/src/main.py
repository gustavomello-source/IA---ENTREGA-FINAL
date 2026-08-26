import random

from src.configuration.config_manager import ConfigManager
from src.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from src.genetic_algorithm.individual import Individual
from src.genetic_algorithm.operators.crossover import OrderCrossover
from src.genetic_algorithm.operators.mutation import SwapMutation
from src.genetic_algorithm.operators.selection import TournamentSelection
from src.tsp.tsp_instance import TSPInstance
from src.tsp.tsp_parser import TSPParser

if __name__ == "__main__":
    config = ConfigManager("config/config.ini")

    instance: TSPInstance = TSPParser.read(path=config.dataset)

    rng = random.Random(config.random_seed)

    selection = TournamentSelection(
        tournament_size=config.tournament_size,
        rng=rng,
    )
    crossover = OrderCrossover(
        crossover_rate=config.crossover_rate,
        rng=rng,
    )
    mutation = SwapMutation(
        mutation_rate=config.mutation_rate,
        rng=rng,
    )

    algorithm = GeneticAlgorithm(
        instance=instance,
        population_size=config.population_size,
        generations=config.generations,
        elitism_size=config.elitism_size,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        rng=rng,
    )

    best: Individual = algorithm.run()

    print(f"Best tour    : {best.tour}")
    print(f"Best distance: {best.distance}")
