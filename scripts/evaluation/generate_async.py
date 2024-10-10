"""
File to be run to generate the search results, and time taken for each query.
"""
import sys
from typing import List, Callable
import numpy as np

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.evaultation.generator_async import AGenerator
from utils.load_query import load_query
from algo.algo_interface import IAlgo
from algo.LSHAlgo import LSH
from algo.HNSWAlgo import HNSW
from algo.VSMAlgo import VSM

import asyncio


async def main():
    # Lazily instantiated list of runners
    runners: List[Callable[[], IAlgo]] = []

    print("Adding runners to list")

    # runners.append(
    #     lambda: VSM("airline_reviews", "bge")
    # )

    m_vals = [2**i for i in range(4, 10)]
    con_vals = [2**i for i in range(10)]
    search_vals = [2**i for i in range(10)]

    # m_vals = [i for i in range(16,16*11+1,16)]
    # con_vals = [i for i in range(30,10*10+30+1,10)]
    # search_vals = [i for i in range(10,5*10+10+1,5)]

    for m in m_vals:
        for c in con_vals:
            for s in search_vals:
                runners.append(
                    lambda m_=m, c_=c, s_=s: HNSW("airline_reviews", "bge", m_, c_, s_)
                )

    # for n_pwr in range(1, 10):
    #     runners.append(lambda pwr=n_pwr: LSH("airline_reviews", "bge", 2**pwr))

    # runners.append(lambda pwr=n_pwr: LSH("airline_reviews", "bge", 768))

    print(f"Starting generator with {len(runners)} runners!")

    # Load questions
    questions = load_query("airline_reviews")
    
    # Execute each runner
    await AGenerator(questions, top_k=10+1).run(runners)


if __name__ == "__main__":
    asyncio.run(main())
