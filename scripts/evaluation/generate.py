"""
File to be run to generate the search results, and time taken for each query.
"""
import sys
from typing import List, Callable
import numpy as np


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

    for n_pwr in range(3, 12):
        runners.append(lambda pwr=n_pwr: LSH("starbucks", "bge", 2**pwr))

    print(f"Starting generator with {len(runners)} runners!")

    # Execute each runner
    Generator([
        "latte",
        "I like sour coffee",
        "Caffeine helps me start my day",
        "Coffeebean is better than starbucks",
        "Water is more worth it than buying coffee from here"
    ]).run(runners)


if __name__ == "__main__":
    main()
