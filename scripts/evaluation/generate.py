"""
File to be run to generate the search results, and time taken for each query.
"""
import sys
from typing import List, Callable


sys.path.append("../..")
from utils.evaultation.generator import Generator
from algo.algo_interface import IAlgo
from algo.VSMAlgo import VSM


def main():

    ### EXAMPLE

    # Lazily instantiated list of runners
    runners:List[Callable[[], IAlgo]] = [
        lambda: VSM("starbucks", "bge", mode="cosine_similarity")
    ]

    # Execute each runner
    Generator(["latte"]).run(runners)





if __name__ == "__main__":
    main()

