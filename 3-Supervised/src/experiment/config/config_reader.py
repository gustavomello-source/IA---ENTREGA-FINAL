import configparser
from typing import Any


class ConfigReader:
    """
    Read config.ini file and return a dictionary with the configuration values.

    Attributes:
        config (configparser.ConfigParser): ConfigParser instance to read the config file.
    """

    def __init__(self) -> None:
        """
        Initialize the ConfigReader instance.
        """
        self.config = configparser.ConfigParser()

    def read_config(self, config_file_path: str) -> dict[Any, Any]:
        """
        Read the config.ini file and return a dictionary with the configuration values.

        Args:
            config_file_path (str): Path to the config.ini file.
        Returns:
            dict[Any, Any]: Dictionary containing the configuration values.
        Raises:
            Exception: If the config file cannot be read.
        """
        try:
            self.config.read(config_file_path)

            config_dict: dict[Any, Any] = {}
            for section in self.config.sections():
                for key, value in self.config.items(section):
                    config_dict[key] = value

            return config_dict
        except Exception as e:
            print(f"Error reading config file: {e}")
            raise

    def get_section(self, section: str) -> dict[str, str]:
        """
        Return the key/value pairs of a single configuration section.

        Args:
            section (str): Name of the configuration section to read.

        Returns:
            dict[str, str]: Mapping of option names to their raw string values.
            An empty dictionary is returned when the section is absent.
        """
        if not self.config.has_section(section):
            return {}
        return dict(self.config.items(section))
