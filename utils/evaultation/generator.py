import sys

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
        self.folder_time = time.strftime("%Y-%M-%D_%H-%M-%S", time.localtime())


    def run(self, runners:List[Callable[[], IAlgo]]) -> None:
        total_queries = len(self.questions) * len(runners)

        with alive_progress.alive_bar(total_queries) as bar:
            for r in runners:
                algo = r()

                res = {
                    "question": [],
                    "top_k": [],
                    "time_taken": []
                }

                for q in self.questions:

                    start_time = time.time_ns()
                    result = algo.run(q, self.k)
                    end_time = time.time_ns()

                    res["question"].append(q)
                    res["top_k"].append(result)
                    res["time_taken"].append((end_time-start_time)//1_000_000)  # convert ns to ms

                    bar()

                res_df = pd.DataFrame.from_dict(res)
                dump_eval_result(self.folder_time, algo.name(), algo.data_source(), res_df, **algo.details())

