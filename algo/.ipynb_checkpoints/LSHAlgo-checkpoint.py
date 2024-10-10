import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Union, Literal, Dict
sys.path.append("..")
import pandas as dd
import numpy as np
from algo.algo_interface import IAlgo
from utils.evaultation.algo_types import AlgoType
from utils.load_data import load_parquet, load_csv, load_index
from sklearn.metrics.pairwise import cosine_similarity
from utils.embedder import Embedder


class LSH(IAlgo):
    def __init__(self, dataset:Literal["starbucks"], embedding, nbits):
        self.embedding_type = embedding
        self.nbits = nbits
        try:
            if self.embedding_type == "bge":
                self.vectorizer = Embedder()
        except:
            raise ValueError("Currently only accept BGE embedding")

        self.data_set_name = dataset
        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/{embedding}.parquet")
        lsh_indexing_file_path = os.path.join(base_dir, f"../data/{dataset}/indexing/lsh{nbits}.index")
        self.data = load_csv(raw_file_path)
        self.embeddings = np.vstack(load_parquet(embedding_file_path).values)
        self.lsh_index = load_index(lsh_indexing_file_path)

        self.mode = "lsh_similarity"
        if self.mode=="lsh_similarity":
            self.method = self.top_k_lsh_similarity

    
    def top_k_lsh_similarity(self, query, k):
        try:
            if self.embedding_type == "bge":
                query_embedded = self.vectorizer.embed(query)
        except:
            raise ValueError("Currently only accept BGE embedding")
        distances, indices = self.lsh_index.search(query_embedded, k)
        top_k_documents = self.data.iloc[indices.flatten()]
        return top_k_documents
        

    def run(self, query, k):
        return self.method(query, k)

    def details(self) -> Dict[str, Union[str, int]]:
        return {
            "embedding": self.embedding_type,
            "mode": self.mode,
            "nbits":self.nbits
        }

    def name(self) -> AlgoType:
        return AlgoType.LSH

    def data_source(self) -> str:
        return self.data_set_name



if __name__ == "__main__":
    lsh = LSH("starbucks", "bge", 2)
    print(lsh.top_k_lsh_similarity("latte", 5))