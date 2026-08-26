from configparser import ConfigParser
from pathlib import Path


class ConfigManager:
    """
    Class to load and validate the project configuration.
    """

    def __init__(self, config_path: str | Path) -> None:
        """
        Initialize the ConfigManager with the path to the
        configuration file.
        """
        self._parser = ConfigParser()

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self._parser.read(config_path)

    def _get_positive_int(
        self, section: str, option: str, *, allow_zero: bool = False
    ) -> int:
        """
        Ensure that the value is a positive integer or zero
        if allowed.

        Args:
            section (str): The section in the configuration file.
            option (str): The option within the section.
            allow_zero (bool): Whether to allow zero as a valid value.
        Returns:
            int: The positive integer value from the configuration.
        Raises:
            ValueError: If the value is not a positive integer or zero (if allowed).
        """
        value: int = self._parser.getint(section, option)

        if allow_zero:
            if value < 0:
                raise ValueError(f"'{option}' must be greater than or equal to 0.")
        else:
            if value <= 0:
                raise ValueError(f"'{option}' must be greater than 0.")

        return value

    def _get_probability(self, section: str, option: str) -> float:
        """
        Ensure that the value is a probability (between 0.0 and 1.0).

        Args:
            section (str): The section in the configuration file.
            option (str): The option within the section.

        Returns:
            float: The probability value from the configuration.
        Raises:
            ValueError: If the value is not a valid probability.
        """
        value: float = self._parser.getfloat(section, option)

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"'{option}' must be between 0.0 and 1.0.")

        return value

    @property
    def population_size(self) -> int:
        return self._get_positive_int(
            "GENETIC_ALGORITHM",
            "population_size",
        )

    @property
    def generations(self) -> int:
        return self._get_positive_int(
            "GENETIC_ALGORITHM",
            "generations",
        )

    @property
    def mutation_rate(self) -> float:
        return self._get_probability(
            "GENETIC_ALGORITHM",
            "mutation_rate",
        )

    @property
    def crossover_rate(self) -> float:
        return self._get_probability(
            "GENETIC_ALGORITHM",
            "crossover_rate",
        )

    @property
    def elitism_size(self) -> int:
        return self._get_positive_int(
            "GENETIC_ALGORITHM",
            "elitism_size",
            allow_zero=True,
        )

    @property
    def tournament_size(self) -> int:
        return self._get_positive_int(
            "GENETIC_ALGORITHM",
            "tournament_size",
        )

    @property
    def random_seed(self) -> int:
        return self._get_positive_int(
            "GENETIC_ALGORITHM",
            "random_seed",
            allow_zero=True,
        )

    @property
    def dataset(self) -> str:
        return self._parser.get(
            "TSP",
            "dataset",
        )

    @property
    def tour_file(self) -> str:
        return self._parser.get(
            "TSP",
            "tour_file",
        )
