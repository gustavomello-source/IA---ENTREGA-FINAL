import argparse
from logging import warning

from nqueens import NQueensSolver
from read_input import FileReader


def main() -> None:
    """
    Call N-Queens solver using the board dimension read from the input file.
    """
    parser = argparse.ArgumentParser(description="Solve the N-Queens problem.")

    parser.add_argument(
        "-f",
        "--file_path",
        dest="file_path",
        required=False,
        help="Path to the input file.",
    )

    args = parser.parse_args()

    if not args.file_path:
        warning("No file path provided.\nTrying to use ./input.txt.")
        args.file_path = "./input.txt"

    reader = FileReader(args.file_path)
    matrix_dimension: int = reader.get_matrix_dimensions()

    solver = NQueensSolver(matrix_dimension)

    print(f"Solver: {solver}\n")

    solution, elapsed_time = solver.solve()

    print("Solution:\n")

    for row in solution:
        print(" ".join("Q" if cell else "." for cell in row))

    print(f"\nIterations: {solver.n_iterations}")
    print(f"Execution time: {elapsed_time:.6f} seconds")


if __name__ == "__main__":
    main()
