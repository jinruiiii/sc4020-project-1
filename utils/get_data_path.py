import os


def get_data_path(name: str) -> str:
    """
    Function to resolve the absolute path to "/data/"
    :return: String, absolute path to file or folder
    """

    name = name.lstrip("/")
    #add folders found in name and check if it exists if not create
    dirs = name.split("/")[:-1]
    path = f"{os.path.dirname(__file__)}/../data/" + "/".join(dirs)
    if not os.path.exists(path):
        os.makedirs(path)

    return f"{os.path.dirname(__file__)}/../data/{name}"
