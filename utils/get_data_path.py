import os

def get_data_path(name: str) -> str:
    """
    Function to resolve the absolute path to "/data/"
    :return: String, absolute path to file or folder
    """

    name = name.lstrip("/")
    #add folders found in name and check if it exists if not create
    file_name = name.split("/")[-1]
    dirs = name.split("/")[:-1]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    path = os.path.join(root, "data")
    for dir in dirs:
        path = os.path.join(path, dir)
    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, file_name)


def get_hnsw_path(dataset: str, time_folder: str, neighbors: int, construction: int, search: int, algo_type: str = "AlgoType.HNSW") -> str:
    """
    Function to get the path for HNSW parquet files
    :param time_folder: e.g. "2024-09-16_06-03-58"
    :param neighbors: number of neighbours used in the runner
    :param construction: corresponding to efConstruction
    :param search: corresponding to efSearch
    :return: absolute file path to the datafile
    """
    file = f"{dataset}_{algo_type}__embedding=bge_mode=hnsw_neighbors={neighbors}_efConstruction={construction}_efSearch={search}.parquet"
    path = get_data_path(f"eval/{time_folder}/{file}")
    assert os.path.exists(path), f"File does not exist! {path}"
    return path


def get_lsh_path(dataset: str, time_folder: str, nbits: int) -> str:
    """
    Function
    :param time_folder: e.g. "2024-09-16_06-03-58"
    :param nbits: number of bits used in LSH runner
    :return: absolute file path to the datafile
    """
    file = f"{dataset}_AlgoType.LSH__embedding=bge_mode=hnsw_similarity_nbits={nbits}.parquet"
    path = get_data_path(f"eval/{time_folder}/{file}")
    assert os.path.exists(path), f"File does not exist! {path}"
    return path
