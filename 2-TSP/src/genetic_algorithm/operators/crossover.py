from __future__ import annotations

import random


class OrderCrossover:
    """
    Order Crossover (OX) operator for permutation-based encodings.

    A contiguous slice of the first parent is copied into the child, and the
    remaining positions are filled with the cities from the second parent in
    the order they appear, skipping cities already present.
    """

    def __init__(self, crossover_rate: float, rng: random.Random) -> None:
        """
        Initialize the order crossover operator.

        Args:
            crossover_rate (float): The probability of applying crossover.
            rng (random.Random): The seeded random number generator.
        """
        self._crossover_rate = crossover_rate
        self._rng = rng

    def crossover(
        self, parent1: list[int], parent2: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Produce two children from two parent tours.

        Args:
            parent1 (list[int]): The first parent tour.
            parent2 (list[int]): The second parent tour.
        Returns:
            tuple[list[int], list[int]]: The two offspring tours.
        """
        if self._rng.random() > self._crossover_rate:
            return parent1[:], parent2[:]

        size: int = len(parent1)

        start: int = self._rng.randint(0, size - 1)
        end: int = self._rng.randint(0, size - 1)

        if start > end:
            start, end = end, start

        child1: list[int] = self._build_child(parent1, parent2, start, end)
        child2: list[int] = self._build_child(parent2, parent1, start, end)

        return child1, child2

    def _build_child(
        self,
        primary: list[int],
        secondary: list[int],
        start: int,
        end: int,
    ) -> list[int]:
        """
        Build a single child using the OX rule.

        Args:
            primary (list[int]): The parent providing the preserved slice.
            secondary (list[int]): The parent providing the remaining order.
            start (int): The inclusive start index of the preserved slice.
            end (int): The inclusive end index of the preserved slice.
        Returns:
            list[int]: The offspring tour.
        """
        size: int = len(primary)

        child: list[int] = [-1] * size
        child[start : end + 1] = primary[start : end + 1]

        preserved: set[int] = set(child[start : end + 1])

        position: int = (end + 1) % size

        for offset in range(size):
            city: int = secondary[(end + 1 + offset) % size]

            if city in preserved:
                continue

            child[position] = city
            position = (position + 1) % size

        return child
