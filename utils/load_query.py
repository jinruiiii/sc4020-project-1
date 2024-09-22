import pandas as pd
from typing import List, Literal
from utils import load_data
from utils.get_data_path import get_data_path


def load_query(dataset:Literal["starbucks", "airline_reviews"]) -> List[str]:
    qn_df = load_data.load_parquet(get_data_path(f"/eval/gt/{dataset}_vsm__type=gt.parquet"))
    return qn_df['question'].tolist()
