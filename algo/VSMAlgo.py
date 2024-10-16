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


class VSM(IAlgo):
    def __init__(self, dataset:Literal["starbucks", "airline_reviews"], embedding):
        self.embedding_type = embedding
        try:
            if self.embedding_type == "bge":
                self.vectorizer = Embedder()
        except:
            raise ValueError("Currently only accept BGE embedding")

        self.data_set_name = dataset
        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/{embedding}.parquet")
        self.data = load_csv(raw_file_path)
        self.embeddings = np.vstack(load_parquet(embedding_file_path).values)
        print(len(self.embeddings))

        self.mode = "cosine_similarity"
        if self.mode=="cosine_similarity":
            self.method = self.top_k_cosine_similarity


    def top_k_cosine_similarity(self, query_embedded, k):
        # try:
        #     if self.embedding_type == "bge":
        #         query_embedded = self.vectorizer.embed(query)
        # except:
        #     raise ValueError("Currently only accept BGE embedding")
        similarities = cosine_similarity(query_embedded, self.embeddings)

        # get top k most similar documents
        try:
            top_k_indices = similarities.argsort()[0][-k:][::-1]
        except IndexError as e:
            print(f"IndexError: {e}. Check if 'similarities' has the expected shape and 'k' is within the valid range.")
            raise


        top_k_documents = self.data.iloc[top_k_indices]

        return top_k_documents, top_k_indices
        

    def run(self, query, k):
        return self.method(query, k)

    def details(self) -> Dict[str, Union[str, int]]:
        return {
            "embedding": self.embedding_type,
            "mode": self.mode,
        }

    def name(self) -> AlgoType:
        return AlgoType.VSM

    def data_source(self) -> str:
        return self.data_set_name



if __name__ == "__main__":
    vsm = VSM("starbucks", "bge")
    print(vsm.top_k_lsh_similarity("latte", 5))