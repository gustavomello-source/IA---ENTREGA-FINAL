class FileReader:
    """
    Class to read a matrix from a text file.
    The first line of the file should contain an integer n, representing
    the dimensions of an n x n matrix.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the FileReader with the path to the text file.

        Args:
            file_path (str): The path to the text file containing the matrix.
        """
        self._file_path = file_path

    def get_matrix_dimensions(self) -> int:
        """
        Read the first line of the file to get the dimensions of the matrix.

        Returns:
            int: The dimension n of the n x n matrix.
        """
        with open(self._file_path, "r") as file:
            first_line = file.readline().strip()
            if not first_line.isdigit():
                raise ValueError(
                    f"The first line of the file must be an integer representing the matrix dimension. Found: {first_line}"
                )
            return int(first_line)

    @property
    def file_path(self) -> str:
        """
        Return the file path for the FileReader instance.
        """
        return self._file_path

    @file_path.setter
    def file_path(self, new_path: str) -> None:
        """
        Set the file path for the FileReader instance. Validate that the new path points to a readable text file.
        """
        with open(new_path, "r") as file:
            if not file.readable():
                raise ValueError(f"The file at {new_path} is not readable.")
            elif not new_path.endswith(".txt"):
                raise ValueError(f"The file at {new_path} is not a text file.")
        self._file_path = new_path
