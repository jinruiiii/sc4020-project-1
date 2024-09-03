import pandas as pd
import os
import sys


sys.path.append("../..")

from typing import List, Tuple
from utils.load_data import load_csv
from utils.evaultation.algo_types import AlgoType
from utils.evaultation.dump_eval_result import dump_eval_result
from algo.VSMAlgo import VSM



class Generate_GT:
    def __init__(self, dataset:str, embedding:str):
        self.VSM = VSM(dataset, embedding)
        base_dir = os.path.dirname(__file__)
        self.dataset = load_csv(os.path.join(base_dir, f"../../data/{dataset}/raw_data/{dataset}.csv"))
        self.questions, self.index_rows = self.generate_qns(32)
        self.answers, self.time_taken = self.generate_top_k(10)
        self.results_df = self.generate_df(self.index_rows, self.answers, self.time_taken)

    def generate_qns(self, sample_size:int) -> Tuple[List[str], List[int]]:
        rows = self.dataset.sample(n=sample_size)
        self.questions = rows["Review"].values.tolist()
        self.index_rows = rows.index.values.tolist()

        return self.questions, self.index_rows
    
    def generate_top_k(self, k:int) -> Tuple[List[List[str]], List[float]]:
        answers = []
        time_taken = []
        for qn in self.questions:
            answer = self.VSM.top_k_cosine_similarity(qn,k)
            answers.append(answer["Review"].values)

        return answers, time_taken

    @staticmethod
    def generate_df(index_rows, answers, time_taken) -> pd.DataFrame:
        df_dict = {
            'q_id': index_rows,
            'top_k': answers,
            'time_taken': time_taken
        }
        df = pd.DataFrame.from_dict(df_dict)
        return df




if __name__ == "__main__":
    generate = Generate_GT("starbucks","bge")
    dump_eval_result(AlgoType.VSM, generate.results_df, type="gt")
