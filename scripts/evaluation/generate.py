"""
File to be run to generate the search results, and time taken for each query.
"""
import sys
from typing import List, Callable
import numpy as np


sys.path.append("../..")
from utils.evaultation.generator import Generator
from algo.algo_interface import IAlgo
from algo.HNSWAlgo import HNSW
from algo.VSMAlgo import VSM


def main():
    # Lazily instantiated list of runners
    runners: List[Callable[[], IAlgo]] = [
        lambda: VSM("starbucks", "bge", mode="cosine_similarity"),
    ]

    for i in range(1, 10):
        # Covers 2^[0, 9] => 1, 2, 4, 8, ..., 512 neighbours
        runners.append(lambda pwr=i: HNSW("starbucks", "bge", 2**pwr))
        print("HNSW with M = ", 2**i)

    for m_pwr in range(10):
        for construct_pwr in range(10):
            for search_pwr in range(10):
                runners.append(lambda m=2**m_pwr, c=2**construct_pwr, s=2**search_pwr: HNSW("starbucks", "bge", m, c, s))

    print(runners)


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
