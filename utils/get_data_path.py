import os


def get_data_path(name: str) -> str:
    """
    Function to resolve the absolute path to "/data/"
    :return: String, absolute path to file or folder
    """

    name = name.lstrip("/")
    return f"{os.path.dirname(__file__)}/../data/{name}"
