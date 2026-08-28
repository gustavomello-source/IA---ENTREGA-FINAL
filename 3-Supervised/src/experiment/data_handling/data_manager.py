from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class DataManager:
    """
    Class responsible for managing the data used in the experiments.
    It provides methods to load, preprocess, and split the data into training and testing sets.

    Attributes:
        data_path (str): Path to the dataset file.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        data (pd.DataFrame): Loaded dataset.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
    """

    def __init__(
        self, data_path: str, test_size: float = 0.2, random_state: int = 1
    ) -> None:
        """
        Initialize the DataManager with the dataset path and parameters for splitting.

        Args:
            data_path (str): Path to the dataset file.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = pd.DataFrame()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> None:
        """
        Load the dataset from the specified path into a pandas DataFrame.

        Raises:
            Exception: If the dataset cannot be loaded.
        """
        try:
            self.data: pd.DataFrame = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def clear_raw_data(self) -> None:
        """
        Clear the raw loaded dataset from memory to free up resources.
        """
        self.data = pd.DataFrame()

    def split_data(self, target_column: str) -> None:
        """
        Split the dataset into training and testing sets, stratified by the target variable.

        Args:
            target_column (str): Name of the target variable column.
        Raises:
            Exception: If the data cannot be split.
        """
        try:
            X: pd.DataFrame = self.data.drop(columns=[target_column])
            y: pd.Series = self.data[target_column]

            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            ) = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
        except Exception as e:
            print(f"Error splitting data: {e}")
            raise

    def save_data_ids(self, output_path: Path) -> None:
        """
        Write the IDs of the training and testing sets to CSV files.

        Args:
            output_path (str): Path to the directory where the ID files will be saved.
        Raises:
            Exception: If the ID files cannot be saved.
        """
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            train_ids = pd.DataFrame(self.X_train.index, columns=["id"])
            test_ids = pd.DataFrame(self.X_test.index, columns=["id"])

            train_ids.to_csv(output_dir / "train_ids.csv", index=False)
            test_ids.to_csv(output_dir / "test_ids.csv", index=False)
        except Exception as e:
            print(f"Error saving data IDs: {e}")
            raise

    def remove_id_column(self, id_column: str) -> None:
        """
        Remove the ID column from the dataset.

        Args:
            id_column (str): Name of the ID column to be removed.
        Raises:
            Exception: If the ID column cannot be removed.
        """
        try:
            if id_column in self.data.columns:
                self.data.drop(columns=[id_column], inplace=True)
        except Exception as e:
            print(f"Error removing ID column: {e}")
            raise
