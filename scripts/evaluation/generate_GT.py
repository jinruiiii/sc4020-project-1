import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import List, Tuple
from utils.load_data import load_csv, load_parquet
from utils.evaultation.algo_types import AlgoType
from utils.evaultation.dump_eval_result import dump_eval_result
from algo.VSMAlgo import VSM
import alive_progress


class Generate_GT:
    def __init__(self, dataset:str, embedding:str, num_questions:int):
        self.VSM = VSM(dataset, embedding)
        base_dir = os.path.dirname(__file__)
        self.dataset_name = dataset
        self.dataset = load_csv(os.path.join(base_dir, f"../../data/{dataset}/raw_data/{dataset}.csv"))
        embedding_file_path = os.path.join(base_dir, f"../../data/{dataset}/embeddings/{embedding}.parquet")
        print(embedding_file_path)
        self.embeddings = np.vstack(load_parquet(embedding_file_path).values)
        print(self.embeddings)
        self.questions, self.index_rows = self.generate_qns(num_questions)

        # 30 + 1 because dont want the query itself to count as the relevant documents
        self.answers, self.time_taken, self.embedded_questions = self.generate_top_k(30+1)
        self.results_df = self.generate_df(questions=self.questions, embedded_questions=self.embedded_questions, answers=self.answers, time_taken=self.time_taken)

    def generate_qns(self, sample_size:int) -> Tuple[List[str], List[int]]:
        rows = self.dataset.sample(n=sample_size)
        if self.dataset_name == "starbucks":
            self.questions = rows["Review"].values.tolist()
        elif self.dataset_name == "airline_reviews":
            self.questions = rows["ReviewBody"].values.tolist()
        self.index_rows = rows.index.values.tolist()

        return self.questions, self.index_rows
    
    def generate_top_k(self, k:int) -> Tuple[List[List[str]], List[float]]:
        answers = []
        time_taken = []
        embedded_questions = []
        with alive_progress.alive_bar(len(self.questions)) as bar:
            for index in self.index_rows:
                print(f"index {index}")
                start_time = time.time_ns()
                embedded_question = self.embeddings[index]
                embedded_questions.append(embedded_question)
                answer, indices = self.VSM.top_k_cosine_similarity(embedded_question.reshape(1, -1),k)
                end_time = time.time_ns()
                time_taken.append((end_time-start_time)//1_000_000)  # convert ns to ms
                if self.dataset_name == "starbucks":
                    answers.append(answer["Review"].values)
                elif self.dataset_name == "airline_reviews":
                    answers.append(answer["ReviewBody"].values)
                bar()

        return answers, time_taken, embedded_questions

    @staticmethod
    def generate_df(questions, embedded_questions, answers, time_taken) -> pd.DataFrame:
        df_dict = {
            'question': questions,
            'embedded_question': embedded_questions,
            'top_k': answers,
            'time_taken': time_taken
        }
        df = pd.DataFrame.from_dict(df_dict)
        return df


if __name__ == "__main__":
    generate = Generate_GT("airline_reviews","bge",32)
    dump_eval_result("gt", AlgoType.VSM, "airline_reviews", generate.results_df, type="gt")
