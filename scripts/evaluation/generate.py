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
    runners: List[Callable[[], IAlgo]] = [
        lambda: VSM("starbucks", "bge"),
    ]

    for m_pwr in range(10):
        for construct_pwr in range(3, 8):
            for search_pwr in range(3, 8):
                runners.append(
                    lambda m=2 ** m_pwr, c=2 ** construct_pwr, s=2 ** search_pwr: HNSW("starbucks", "bge", m, c, s))

    for n_pwr in range(3, 10):
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
