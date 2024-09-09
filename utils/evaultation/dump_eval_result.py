from typing import Literal

import pandas as pd

from utils.evaultation.algo_types import AlgoType
from utils.get_data_path import get_data_path


def dump_eval_result(
        folder_name: str,
        algo_name: AlgoType,
        dataset_name: Literal["starbucks"],
        data: pd.DataFrame,
        **kwargs
) -> None:
    """
    Method to save pandas df to file.
    :param folder_name: Name of subfolder for the files to be placed in
    :param algo_name: Name of the algorithm used
    :param dataset_name: Specifies the name of the dataset used
    :param data: Pandas data frame with the following columns and types
        q_id: int; search index number
        top_k: List[int]; array id corresponding to the results and the order of results.
        time_taken: float: time taken to perform the search
    :param kwargs: additional keywords to be added to the file name
    :return:
    """

    attr_names = "_".join([f"{k}={v}" for k, v in kwargs.items()])
    save_path = f"eval/{folder_name}/{dataset_name}_{algo_name}__{attr_names}.parquet"
    
    data.to_parquet(get_data_path(save_path))
