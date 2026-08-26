from pathlib import Path


class TourParser:
    """
    Class to read a tour from a file.
    """

    @staticmethod
    def read(path: str | Path) -> list[int]:
        """
        Reads a tour from a file.

        Args:
            path (str | Path): The path to the file containing the tour.

        Returns:
            list[int]: A list of city indices representing the tour.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        tour: list[int] = []
        reading = False

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                if line == "TOUR_SECTION":
                    reading = True
                    continue

                if not reading:
                    continue

                if line == "EOF":
                    break

                node: int = int(line)

                if node == -1:
                    break

                tour.append(node - 1)

        return tour
