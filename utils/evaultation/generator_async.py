import sys
import json

from utils.evaultation.dump_eval_result import dump_eval_result

sys.path.append("../..")
import alive_progress
from typing import List, Callable
import time
import pandas as pd

from algo.algo_interface import IAlgo

import asyncio
from tqdm.asyncio import tqdm

class AGenerator:

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

    async def query(self, algo, q):
        # print(f"Query: {q}")
        print("starting query")
        start_time = time.time_ns()
        result = algo.run(q, self.k)
        end_time = time.time_ns()
        print(f"time taken for a single question for {algo.name()}, {algo.details()}: {(end_time - start_time) // 1_000_000}")
        time_taken = (end_time - start_time) // 1_000_000
        return [q, result, time_taken]

    async def run(self, runners: List[Callable[[], IAlgo]]) -> None:
        
        total_queries = len(self.questions) * len(runners)

        # with alive_progress.alive_bar(total_queries) as bar:
        for r in runners:
            start = time.time_ns()
            print("Creating next runner!")
            algo = r()
            print(f"Starting runner for {algo.name()}, {algo.details()}")

            # res = {
            #     "question": [],
            #     "top_k": [],
            #     "time_taken": []
            # }

            # for q in self.questions:
            #     print(f"Query: {q}")
            #     start_time = time.time_ns()
            #     result = algo.run(q, self.k)
            #     end_time = time.time_ns()

            #     res["question"].append(q)
            #     res["top_k"].append(result)
            #     res["time_taken"].append((end_time - start_time) // 1_000_000)  # convert ns to ms

            #     print(f"time taken for a single question for {algo.name()}, {algo.details()}: {(end_time - start_time) // 1_000_000}")
            #     run_time += (end_time - start_time) // 1_000_000
            #     bar()
            
            coros = [self.query(algo,q) for q in self.questions]
            results = await tqdm.gather(*coros)

            column_names = ["question", "top_k", "time_taken"]
            res_df = pd.DataFrame(results, columns=column_names)
            res_df['top_k'] = res_df["top_k"].apply(lambda x: json.dumps(x.to_dict()))
            print(res_df)
            end = time.time_ns()
            run_time = end-start // 1_000_000
            print(f"time taken for a run for {algo.name()}, {algo.details()}: {run_time} ms")

            dump_eval_result(self.folder_time, algo.name(), algo.data_source(), res_df, **algo.details())
