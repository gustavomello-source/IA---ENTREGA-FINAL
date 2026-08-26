from typing import Any

from src.experiment.config.config_reader import ConfigReader
from src.experiment.context import PipelineContext
from src.experiment.data.data_manager import DataManager
from src.experiment.experiment import Experiment
from src.utils.logging_utils import setup_logging
from src.utils.remove_temp_files import remove_temp_files


def main() -> None:

    config_reader = ConfigReader()
    config: dict[Any, Any] = config_reader.read_config("./config.ini")
    print("Configuration values:", config)

    logger, experiment_folder = setup_logging(
        report_path=config.get("LOG", {}).get("report_path", "./report/")
    )

    data_manager = DataManager(
        data_path=config.get("DATA", {}).get("data_path", "./data/raw/dataset.csv")
    )
    data_manager.load_data()

    data_manager.split_data(
        target_column=config.get("DATA", {}).get("target_column", "Y")
    )
    data_manager.save_data_ids(
        output_path=config.get("DATA", {}).get(
            "data_ids_output_path", "./data/processed/data_ids.csv"
        )
    )
    data_manager.clear_raw_data()

    context = PipelineContext(
        config=config_reader,
        logger=logger,
        data_manager=data_manager,
        experiment_folder=experiment_folder,
    )

    experiment = Experiment(context)
    experiment.run()

    # end of experiment
    logger.info("Cleaning up temporary files.")
    remove_temp_files(temp_file_names=["__pycache__/"])

    logger.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()
