"""
File to be run to generate the search results, and time taken for each query.
"""
import sys
from typing import List, Callable, Literal
import numpy as np
import argparse

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.evaultation.generator import Generator
from utils.load_query import load_embedded_query
from algo.algo_interface import IAlgo
from algo.LSHAlgo import LSH
from algo.HNSWAlgo import HNSW
from algo.VSMAlgo import VSM


def main(args: argparse.Namespace):
    # Lazily instantiated list of runners
    runners: List[Callable[[], IAlgo]] = []
    dataset = args.dataset
    embedding_type = args.embedding_type

    print("Adding runners to list")

    if args.vsm == True:
        print("adding VSM runners")
        runners.append(
            lambda: VSM(dataset, embedding_type)
        )

    if args.hnsw == True:
        print("adding HNSW runners")
        m_vals = [2**i for i in range(4, 10)]
        con_vals = [i for i in range(50, 501, 50)]
        search_vals = [2**i for i in range(10)]

        for m in m_vals:
            for c in con_vals:
                for s in search_vals:
                    runners.append(
                        lambda m_=m, c_=c, s_=s: HNSW(dataset, embedding_type, m_, c_, s_)
                    )

    if args.lsh == True:
        print("adding LSH runners")
        for n_pwr in range(1, 10):
            runners.append(lambda pwr=n_pwr: LSH(dataset, embedding_type, 2**pwr))

        runners.append(lambda pwr=n_pwr: LSH(dataset, embedding_type, 768))

    print(f"Starting generator with {len(runners)} runners!")

    # Load questions
    print("Loading questions and embeddings")
    questions, embedded_questions = load_embedded_query(dataset, vsm_type=args.vsm_file_type, type=args.gt_type)
    
    # Execute each runner
    if args.is_batch:
        print("Running batch")
        Generator(questions=questions, embedded_questions=embedded_questions, top_k=10+1).run_batch(runners)
    else:
        print("Running individually")
        Generator(questions=questions, embedded_questions=embedded_questions, top_k=10+1).run(runners)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate search results for dataset")
    parser.add_argument('--vsm', action=argparse.BooleanOptionalAction, help="Whether to include VSM")
    parser.add_argument('--hnsw', action=argparse.BooleanOptionalAction, help="Whether to include HNSW")
    parser.add_argument('--lsh', action=argparse.BooleanOptionalAction, help="Whether to include LSH")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to perform search.")
    parser.add_argument('--is_batch', action=argparse.BooleanOptionalAction, help="Whether to search all questions at once or one question at a time")
    parser.add_argument("--embedding_type", type=str, required=False, default="bge", help="Type of embedding")
    parser.add_argument("--vsm_file_type", type=str, required=False, default="vsm", help="Name of VSM in the VSM file")
    parser.add_argument("--gt_type", type=str, required=False, default="gt", help="gt(n) for n number of ground truths generated")
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args=args)
