import time


class NQueensSolver:
    """
    Class to solve the N-Queens problem using recursive backtracking
    and heuristics.
    """

    def __init__(self, n: int):
        """
        Initialize the solver for the N-Queens problem.

        Args:
            n (int): Board dimension.
        """
        self.dimension = n
        self.n_iterations = 0

        self.positions: list[int] = [-1] * n

        self.rows: set[int] = set()
        self.main_diagonals: set[int] = set()
        self.anti_diagonals: set[int] = set()

    def __str__(self) -> str:
        """
        Return a string representation of the solver.
        """
        return f"NQueensSolver(n={self.dimension})"

    def solve(self) -> tuple[list[list[int]], float]:
        """
        Solve the N-Queens problem.

        Returns:
            tuple[list[list[int]], float]:
                The solved board and the execution time.
        """
        start_time = time.time()

        self._solve_n_queens(0)

        board = self._positions_to_board()

        end_time = time.time()

        return board, end_time - start_time

    def _solve_n_queens(self, col: int) -> bool:
        """
        Solve the problem recursively.
        First, it checks if the current column is equal to the dimension of the board.
        If so, it means that all queens have been placed successfully, and the function returns True
        Then, it generates a list of candidate rows for the current column, filtering out rows
        that are already occupied by queens or are in conflict with existing queens based on the row and diagonal constraints.
        The function then iterates through the sorted candidate rows, placing a queen in each row and recursively
        calling itself to attempt to place queens in the next column.

        Args:
            col (int): Current column.

        Returns:
            bool: True if a solution is found.
        """

        self.n_iterations += 1

        if col == self.dimension:
            return True

        candidate_rows: list[tuple[int, int]] = []

        for row in range(self.dimension):
            if (
                row in self.rows
                or (row - col) in self.main_diagonals
                or (row + col) in self.anti_diagonals
            ):
                continue

            score = self._calculate_heuristic_score(row, col)
            candidate_rows.append((score, row))

        candidate_rows.sort()

        for _, row in candidate_rows:
            self.positions[col] = row

            self.rows.add(row)
            self.main_diagonals.add(row - col)
            self.anti_diagonals.add(row + col)

            if self._solve_n_queens(col + 1):
                return True

            self.positions[col] = -1

            self.rows.remove(row)
            self.main_diagonals.remove(row - col)
            self.anti_diagonals.remove(row + col)

        return False

    def _calculate_heuristic_score(self, row: int, col: int) -> int:
        """
        Calculate the heuristic score for placing a queen at (row, col).

        This heuristic estimates how many valid positions remain in the
        next column after placing the queen.

        The lower the score, the better the placement, as it leaves more options
        for future placements.

        Args:
            row (int): Candidate row.
            col (int): Current column.

        Returns:
            int: Heuristic score.
        """

        if col == self.dimension - 1:
            return 0

        self.rows.add(row)
        self.main_diagonals.add(row - col)
        self.anti_diagonals.add(row + col)

        next_col: int = col + 1
        available = 0

        for next_row in range(self.dimension):
            if (
                next_row not in self.rows
                and (next_row - next_col) not in self.main_diagonals
                and (next_row + next_col) not in self.anti_diagonals
            ):
                available += 1

        self.rows.remove(row)
        self.main_diagonals.remove(row - col)
        self.anti_diagonals.remove(row + col)

        return -available

    def _positions_to_board(self) -> list[list[int]]:
        """
        Convert the compact representation into a board matrix.

        Returns:
            list[list[int]]: Board representation.
        """

        board: list[list[int]] = [[0] * self.dimension for _ in range(self.dimension)]

        for col, row in enumerate(self.positions):
            if row != -1:
                board[row][col] = 1

        return board
