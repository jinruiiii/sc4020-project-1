import faiss
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_data import load_parquet, load_csv, load_index
from utils.embedder import Embedder

class HNSW:
    def __init__(self, dataset, embedding_type, M: int) -> None:
        """
        dataset: name of dataset folder
        embedding: name of embeddings to use
        M: no. of neighbours during insertion
        """
        self.embedding_type = embedding_type
        try:
            if self.embedding_type == "bge":
                # self.vectorizer = Embedder()
                self.d = 1024
        except:
            raise ValueError("Currently only accept BGE embedding")
        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/{embedding_type}.parquet")
        # lsh_indexing_file_path = os.path.join(base_dir, f"../data/{dataset}/indexing/lsh.index")
        self.data = load_csv(raw_file_path)
        self.embeddings = np.vstack(load_parquet(embedding_file_path)["embedding"].values)
        # self.lsh_index = load_index(lsh_indexing_file_path)
        print(self.embeddings.shape)
        self.M = M
        self.index = faiss.IndexHNSWFlat(self.d, self.M)
        print("initialized")

    def get_embeddings(self):
        return self.embeddings

    def construct_graph(self, efConstruction, efSearch, M = None, mL = None) -> None:
        """
        efConstruction: no. of nearest neighbours to explore during construction
        efSearch: no. of nearest neighbours to explore during search
        M:
        mL: normalization factor (optional)
        """
        print("in construction")
        self.index.hnsw.efConstruction = efConstruction
        if M is not None and mL is not None:
            self.index.set_default_probas(M, mL)
        self.index.add(self.embeddings)  # build the index
        print("Added")
        self.index.hnsw.efSearch = efSearch

        print("finish construction")
        return self.index

    def search(self, query, k: int):
        """
        query:
        k: no. of nearest neighbours to search 
        """
        print("searching")
        result = self.index.search(query, k)
        return result