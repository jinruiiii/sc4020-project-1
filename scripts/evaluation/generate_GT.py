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

    def generate_qns(self, sample_size:int) -> Tuple[List[str], List[int]]:
        rows = self.dataset.sample(n=sample_size)
        self.questions = rows["Review"].values.tolist()
        self.index_rows = rows.index.values.tolist()

        return self.questions, self.index_rows
    
    def generate_top_k(self, k:int):
        answers = []
        for qn in self.questions:
            answer = self.VSM.top_k_cosine_similarity(qn,k)
            answers.append(answer)

        return answers

if __name__ == "__main__":
    generate = Generate_GT("starbucks","bge")
    print(len(generate.answers), len(generate.answers[0]))