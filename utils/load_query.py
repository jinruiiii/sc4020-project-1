import pandas as pd
from typing import List, Literal

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_data
from utils.get_data_path import get_data_path
from utils.embedder import Embedder

import time


def load_query(dataset:Literal["starbucks", "airline_reviews"], vsm_type:str = "vsm", type:str ="gt") -> List[str]:
    qn_df = load_data.load_parquet(get_data_path(f"/eval/gt/{dataset}_{vsm_type}__type={type}.parquet"))
    return qn_df['question'].tolist()

def load_embedded_query(dataset:Literal["starbucks", "airline_reviews"], vsm_type:str = "vsm", type:str ="gt") -> List[str]:
    qn_df = load_data.load_parquet(get_data_path(f"/eval/gt/{dataset}_{vsm_type}__type={type}.parquet"))
    # print(qn_df)
    questions = qn_df['question'].tolist()
    embedded_questions = qn_df['embedded_question'].tolist()


    # embedded_queries = []
    # start = time.time_ns()
    # vectorizer = Embedder()
    # embedded_queries = vectorizer.batch_embed(questions)
    # end = time.time_ns()
    # print(f"time taken to embed {len(qn_df)} queries: {(end-start)//1_000_000} ms")
    return questions, embedded_questions