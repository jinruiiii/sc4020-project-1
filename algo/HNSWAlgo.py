import os
import pickle
import sys
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Union, Literal, Dict
sys.path.append("..")
import pandas as pd
import numpy as np
from algo.algo_interface import IAlgo
from utils.evaultation.algo_types import AlgoType
from utils.load_data import load_csv, load_parquet
from utils.embedder import Embedder

class HNSW(IAlgo):
    def __init__(self, dataset: Literal["starbucks"], embedding_type: str, M: int):
        """
        dataset: name of dataset folder
        embedding_type: name of embeddings to use
        M: no. of neighbours during insertion
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

        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/{embedding_type}.parquet")
        
        # load raw data
        self.data = load_csv(raw_file_path)

        # load embeddings
        self.data_store_embeddings = np.vstack(load_parquet(embedding_file_path).values).astype('float32')

        self.M = M

        # fetch index
        self.index = None
        self.load_index()

        self.mode = "hnsw"
        self.method = self.search

    def search(self, query: str, k: int):
        """
        query: single query to search for
        k: no. of nearest neighbours to search 

        Returns (list of indices corresponding to nearest neighbours of query, list of data rows corresponding to nearest neighbours of query)
        """
        embedded_queries = []
        try:
            if self.embedding_type == "bge":
                embedded_queries = self.vectorizer.embed(query)
        except:
            raise ValueError("Currently only accept BGE embedding")
        
        D, I = self.index.search(embedded_queries, k)

        # results = [[self.data.iloc[row] for row in result] for result in I]
        results = self.data.iloc[I.flatten()]
        return results

    def load_index(self):
        """
        Load index from index file
        Returns HNSW index
        """
        base_dir = os.path.dirname(__file__)
        index_file_path = os.path.join(base_dir, f"../data/{self.data_set_name}/indexing/hnsw.index")
        self.index = faiss.read_index(index_file_path)
        return
    
    def run(self, query, k):
        return self.method(query, k)
    
    def details(self) -> Dict[str, Union[str, int]]:
        return {
            "embedding": self.embedding_type,
            "mode": self.mode,
            "neighbors": self.M
        }
    
    def name(self) -> AlgoType:
        return AlgoType.HNSW
    
    def data_source(self) -> str:
        return self.data_set_name
    
if __name__ == "__main__":
    algo = HNSW("starbucks", "bge", 5)
    results = algo.run("Latte", 5)
    print(results)
