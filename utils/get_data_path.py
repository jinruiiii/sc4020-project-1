import os
from utils.load_data import load_parquet


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


def get_hnsw_path(time_folder: str, neighbors: int, construction: int, search: int) -> str:
    """
    Function to get the path for HNSW parquet files
    :param time_folder: e.g. "2024-09-16_06-03-58"
    :param neighbors: number of neighbours used in the runner
    :param construction: corresponding to efConstruction
    :param search: corresponding to efSearch
    :return: absolute file path to the datafile
    """
    file = f"starbucks_AlgoType.HNSW__embedding=bge_mode=hnsw_neighbors={neighbors}_efConstruction={construction}_efSearch={search}.parquet"
    path = get_data_path(f"eval/{time_folder}/{file}")
    assert os.path.exists(path), f"File does not exist! {path}"
    return path


def get_lsh_path(time_folder: str, nbits: int) -> str:
    """
    Function
    :param time_folder: e.g. "2024-09-16_06-03-58"
    :param nbits: number of bits used in LSH runner
    :return: absolute file path to the datafile
    """
    file = f"starbucks_AlgoType.LSH__embedding=bge_mode=hnsw_similarity_nbits={nbits}.parquet"
    path = get_data_path(f"eval/{time_folder}/{file}")
    assert os.path.exists(path), f"File does not exist! {path}"
    return path
