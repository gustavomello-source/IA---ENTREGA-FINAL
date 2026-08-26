from pathlib import Path

from src.tsp.city import City
from src.tsp.tsp_instance import TSPInstance


class TSPParser:
    """
    Reads a TSPLIB instance.
    """

    @staticmethod
    def read(path: str | Path) -> TSPInstance:
        """
        Read a TSPLIB instance from a file, iterate through its lines,
        and parse the relevant information to create a TSPInstance object.

        Args:
            path (str | Path): The path to the TSPLIB instance file.
        Returns:
            TSPInstance: The parsed TSP instance.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        name: str = ""
        comment: str = ""
        dimension: int = 0
        edge_weight_type: str = ""

        cities: list[City] = []

        reading_nodes = False

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                if line == "NODE_COORD_SECTION":
                    reading_nodes = True
                    continue

                if reading_nodes:
                    if line == "EOF":
                        break

                    parts = line.split()

                    city = City(
                        id=int(parts[0]),
                        x=float(parts[1]),
                        y=float(parts[2]),
                    )

                    cities.append(city)

                    continue

                if ":" in line:
                    key, value = line.split(":", maxsplit=1)

                    key: str = key.strip()
                    value: str = value.strip()

                    match key:
                        case "NAME":
                            name = value

                        case "COMMENT":
                            comment = value

                        case "DIMENSION":
                            dimension = int(value)

                        case "EDGE_WEIGHT_TYPE":
                            edge_weight_type = value

        if len(cities) != dimension:
            raise ValueError(f"Expected {dimension} cities but found {len(cities)}.")

        return TSPInstance(
            name=name,
            comment=comment,
            dimension=dimension,
            edge_weight_type=edge_weight_type,
            cities=cities,
        )
