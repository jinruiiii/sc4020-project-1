import os
import pickle
import sys
import faiss
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Union, Literal, Dict, Optional, List, Any

sys.path.append("..")
import pandas as pd
import numpy as np
from algo.algo_interface import IAlgo
from utils.evaultation.algo_types import AlgoType
from utils.load_data import load_csv, load_parquet
from utils.embedder import Embedder


class HNSW(IAlgo):
    def __init__(
            self,
            dataset: Literal["starbucks", "airline_reviews"],
            embedding_type: str,
            M: int,
            efConstruction: Optional[int] = None,
            efSearch: Optional[int] = None
    ):
        """
        dataset: name of dataset folder
        embedding_type: name of embeddings to use
        M: no. of neighbours during insertion
        efConstruction: optional, number of nearest neighbours to explore during construction, default: 40
        efSearch: optional, nuber of nearest neighbours to explore during search, default: 16
        """
        self.embedding_type = embedding_type
        try:
            if self.embedding_type == "bge":
                self.vectorizer = Embedder()
                # dimension of embeddings
                self.d = 1024
        except:
            raise ValueError("Currently only accept BGE embedding")

        self.data_set_name = dataset
        self.efConstruction = efConstruction
        self.efSearch = efSearch

        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/{embedding_type}.parquet")

        # load raw data
        self.data = load_csv(raw_file_path)

        # load embeddings
        self.data_store_embeddings = np.vstack(load_parquet(embedding_file_path).values).astype('float32')

        self.M = M

        # fetch index
        self.index = self._construct_graph(self.efConstruction, self.efSearch)

        self.mode = "hnsw"
        self.method = self.search

    def _construct_graph(self, efConstruction=None, efSearch=None, mL=None):
        """
        efConstruction: no. of nearest neighbours to explore during construction (optional)
        efSearch: no. of nearest neighbours to explore during search (optional)
        mL: normalization factor (optional)

        Returns the created HNSW Index
        """
        start = time.time_ns()
        index = faiss.IndexHNSWFlat(self.d, self.M)

        if efConstruction is not None:
            index.hnsw.efConstruction = efConstruction

        index.add(self.data_store_embeddings)  # build the index

        # change efSearch after adding the data
        if efSearch is not None:
            index.hnsw.efSearch = efSearch
        
        end = time.time_ns()
        print(index.hnsw.entry_point)
        print(f"time taken for index construction for HNSW, efConstruction: {efConstruction}, efSearch: {efSearch}: {(end-start)/1_000_000} ns")
        return index

    def search(self, embedded_queries:List[Any], k: int):
        """
        embedded_query: single query to search for (embedded)
        k: no. of nearest neighbours to search 

        Returns (list of indices corresponding to nearest neighbours of query, list of data rows corresponding to nearest neighbours of query)
        """
        # embedded_queries = []
        # try:
        #     if self.embedding_type == "bge":
        #         embedded_queries = self.vectorizer.embed(query)
        # except:
        #     raise ValueError("Currently only accept BGE embedding")

        start_time = time.time_ns()
        D, I = self.index.search(embedded_queries, k)
        end_time = time.time_ns()
        duration = end_time - start_time

        # results = [[self.data.iloc[row] for row in result] for result in I]
        print(I.shape)
        results = self.data.iloc[I.flatten()]
        return results, duration

    def run(self, query, k):
        return self.method(query, k)

    def details(self) -> Dict[str, Union[str, int]]:
        return {
            "embedding": self.embedding_type,
            "mode": self.mode,
            "neighbors": self.M,
            "efConstruction": self.efConstruction,
            "efSearch": self.efSearch
        }

    def name(self) -> AlgoType:
        return AlgoType.HNSW

    def data_source(self) -> str:
        return self.data_set_name


if __name__ == "__main__":
    algo = HNSW("starbucks", "bge", 5)
    results = algo.run("Latte", 5)
    print(results)
