import sys
import json

from utils.evaultation.dump_eval_result import dump_eval_result

sys.path.append("../..")
import alive_progress
from typing import List, Callable
import time
import pandas as pd

from algo.algo_interface import IAlgo


class Generator:

    def __init__(
            self,
            questions: List[str],
            top_k: int = 10
    ):
        """
        :param questions: Questions to query each runner
        :param top_k: Number of top similar chunks
        """
        self.questions = questions
        self.k = top_k
        self.folder_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    def run(self, runners: List[Callable[[], IAlgo]]) -> None:
        total_queries = len(self.questions) * len(runners)

        with alive_progress.alive_bar(total_queries) as bar:
            for r in runners:
                start = time.time_ns()
                print("Creating next runner!")
                algo = r()
                print(f"Starting runner for {algo.name()}, {algo.details()}")

                res = {
                    "question": [],
                    "top_k": [],
                    "time_taken": []
                }

                for q in self.questions:
                    print(f"Query: {q}")
                    # start_time = time.time_ns()
                    result, duration = algo.run(q, self.k)
                    # end_time = time.time_ns()

                    res["question"].append(q)
                    res["top_k"].append(result)
                    res["time_taken"].append(duration // 1_000_000)  # convert ns to ms

                    bar()

                res_df = pd.DataFrame.from_dict(res)
                res_df['top_k'] = res_df["top_k"].apply(lambda x: json.dumps(x.to_dict()))
                end = time.time_ns()
                duration = end-start
                print(f"time taken for runner for {algo.name()}, {algo.details()}: {duration // 1_000_000} ms")
                dump_eval_result(self.folder_time, algo.name(), algo.data_source(), res_df, **algo.details())
