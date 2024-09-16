"""
File to be run to generate the search results, and time taken for each query.
"""
import sys
from typing import List, Callable
import numpy as np

from utils.load_query import load_query

sys.path.append("../..")
from utils.evaultation.generator import Generator
from algo.algo_interface import IAlgo
from algo.LSHAlgo import LSH
from algo.HNSWAlgo import HNSW
from algo.VSMAlgo import VSM


def main():
    # Lazily instantiated list of runners
    runners: List[Callable[[], IAlgo]] = []

    print("Adding runners to list")

    runners.append(
        lambda: VSM("starbucks", "bge")
    )

    m_vals = [2**i for i in range(4, 10)]
    con_vals = [2**i for i in range(10)]
    search_vals = [2**i for i in range(10)]

    for m in m_vals:
        for c in con_vals:
            for s in search_vals:
                runners.append(
                    lambda m_=m, c_=c, s_=s: HNSW("starbucks", "bge", m_, c_, s_)
                )

    for n_pwr in range(1, 10):
        runners.append(lambda pwr=n_pwr: LSH("starbucks", "bge", 2**pwr))

    runners.append(lambda pwr=n_pwr: LSH("starbucks", "bge", 768))

    print(f"Starting generator with {len(runners)} runners!")

    # Load questions
    questions = load_query("starbucks")

    # Execute each runner
    Generator(questions, top_k=10+1).run(runners)


if __name__ == "__main__":
    main()
