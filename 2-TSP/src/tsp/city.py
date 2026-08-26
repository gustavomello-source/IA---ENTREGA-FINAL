from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class City:
    """
    Represents a city in a TSP instance.

    Attributes:
        id (int): The unique identifier of the city.
        x (float): The x-coordinate of the city.
        y (float): The y-coordinate of the city.
    """

    id: int
    x: float
    y: float
