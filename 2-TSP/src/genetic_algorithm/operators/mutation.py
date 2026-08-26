from __future__ import annotations

import random


class SwapMutation:
    """
    Swap mutation operator for permutation-based encodings.
    """

    def __init__(self, mutation_rate: float, rng: random.Random) -> None:
        """
        Initialize the swap mutation operator.

        Args:
            mutation_rate (float): The probability of applying a swap.
            rng (random.Random): The seeded random number generator.
        """
        self._mutation_rate = mutation_rate
        self._rng = rng

    def mutate(self, tour: list[int]) -> list[int]:
        """
        Possibly swap two cities in the given tour.

        Args:
            tour (list[int]): The tour to mutate.
        Returns:
            list[int]: The (possibly) mutated tour.
        """
        mutated: list[int] = tour[:]

        if self._rng.random() > self._mutation_rate:
            return mutated

        i: int = self._rng.randint(0, len(mutated) - 1)
        j: int = self._rng.randint(0, len(mutated) - 1)

        mutated[i], mutated[j] = mutated[j], mutated[i]

        return mutated
