from pathlib import Path


def remove_temp_files(temp_file_names: list[str]) -> bool:
    """
    Recursively remove temporary and artifact files with the same names as those in the provided list,
    in any directory, including subdirectories.

    Args:
        temp_file_names (list[str]): List of temporary file names to remove.
    """
    try:
        for file in temp_file_names:
            for path in Path().rglob(file):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    for sub_path in path.rglob("*"):
                        if sub_path.is_file():
                            sub_path.unlink()
                    path.rmdir()
        return True
    except Exception as e:
        print(f"Error removing temporary files: {e}")
        return False
