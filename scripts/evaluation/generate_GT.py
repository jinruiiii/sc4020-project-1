import pandas as pd
import numpy as np
import os
import sys
sys.path.append("../..")

from typing import List, Tuple
from utils.load_data import load_csv
from scripts.VSMAlgo import VSM

class Generate_GT:
    def __init__(self, dataset:str, embedding:str):
        self.VSM = VSM(dataset, embedding)
        base_dir = os.path.dirname(__file__)
        self.dataset = load_csv(os.path.join(base_dir, f"../../data/{dataset}/raw_data/{dataset}.csv"))
        self.questions, self.index_rows = self.generate_qns(32)
        self.answers = self.generate_top_k(10)
        self.results_df = self.generate_df(self.index_rows, self.answers)

    def generate_qns(self, sample_size:int) -> Tuple[List[str], List[int]]:
        rows = self.dataset.sample(n=sample_size)
        self.questions = rows["Review"].values.tolist()
        self.index_rows = rows.index.values.tolist()

        return self.questions, self.index_rows
    
    def generate_top_k(self, k:int) -> List[List[str]]:
        answers = []
        for qn in self.questions:
            answer = self.VSM.top_k_cosine_similarity(qn,k)
            answers.append(answer["Review"].values)

        return answers
    
    def generate_df(self, index_rows, answers) -> pd.DataFrame:
        df_dict = {}
        df_dict['q_id'] = index_rows
        df_dict['top_k'] = answers
        df = pd.DataFrame.from_dict(df_dict)
        return df
    
if __name__ == "__main__":
    generate = Generate_GT("starbucks","bge")