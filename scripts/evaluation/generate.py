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

    for i in range(3):
        # Covers 2^[0, 9] => 1, 2, 4, 8, ..., 512 neighbours
        print(f"Constructing graph for i={i}")
        hnsw = HNSW("starbucks", "bge", 2**i)
        hnsw.construct_graph(efConstruction=i, efSearch=i)
        runners.append(lambda pwr=i: HNSW("starbucks", "bge", 2**pwr))
        print("HNSW with M = ", 2**i)

    print(len(runners))

    # Execute each runner
    Generator(["latte"]).run(runners)


if __name__ == "__main__":
    main()
