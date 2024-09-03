import pandas as pd

from utils.evaultation.algo_types import AlgoType
from utils.get_data_path import get_data_path


def dump_eval_result(
        algo_name: AlgoType,
        data: pd.DataFrame,
        **kwargs
) -> None:
    """
    Method to save pandas df to file.
    :param algo_name: Name of the algorithm used
    :param data: Pandas data frame with the following columns and types
        q_id: int; search index number
        top_k: List[int]; array id corresponding to the results and the order of results.
        time_taken: float: time taken to perform the search
    :param kwargs: additional keywords to be added to the file name
    :return:
    """

    attr_names = "_".join([f"{k}={v}" for k, v in kwargs.items()])
    data.to_parquet(get_data_path(f"eval/{algo_name}_{attr_names}.parquet"))
