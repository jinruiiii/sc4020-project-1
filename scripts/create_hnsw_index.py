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
        embedding_type: name of embeddings to use
        M: no. of neighbours during insertion
        """
        self.dataset = dataset
        self.embedding_type = embedding_type
        try:
            if self.embedding_type == "bge":
                # initialize vectorizer
                self.vectorizer = Embedder()

                # dimension of embeddings
                self.d = 1024
        except:
            raise ValueError("Currently only accept BGE embedding")
        
        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/{embedding_type}.parquet")
        
        # load raw data
        self.data = load_csv(raw_file_path)

        # load embeddings
        self.data_store_embeddings = np.vstack(load_parquet(embedding_file_path).values).astype('float32')
        
        self.M = M

        # create index
        self.index = faiss.IndexHNSWFlat(self.d, self.M)

    def construct_graph(self, efConstruction=None, efSearch=None, M = None, mL = None) -> None:
        """
        efConstruction: no. of nearest neighbours to explore during construction (optional)
        efSearch: no. of nearest neighbours to explore during search (optional)
        M: no. of neighbours during insertion (optional)
        mL: normalization factor (optional)

        Returns the created HNSW Index
        """
        
        if efConstruction is not None:
            self.index.hnsw.efConstruction = efConstruction

        if M is not None and mL is not None:
            self.index.set_default_probas(M, mL)

        self.index.add(self.data_store_embeddings)  # build the index
        
        # change efSearch after adding data
        if efSearch is not None:
            self.index.hnsw.efSearch = efSearch
        
        return self.index

    def search(self, queries, k: int):
        """
        queries: list of strings of natural language queries
        k: no. of nearest neighbours to search 

        Returns (list of indices corresponding to nearest neighbours of query, list of data rows corresponding to nearest neighbours of query)
        """
        embedded_queries = []
        try:
            if self.embedding_type == "bge":
                embedded_queries = np.array([self.vectorizer.embed(query).squeeze(0).numpy() for query in queries])
        except:
            raise ValueError("Currently only accept BGE embedding")
        
        D, I = self.index.search(embedded_queries, k)

        results = [[self.data.iloc[row] for row in result] for result in I]
        return I, results
    
    def save_index(self):
        """
        Saves built HNSW index into index file
        """
        base_dir = os.path.dirname(__file__)
        index_file_path = os.path.join(base_dir, f"../data/{self.dataset}/indexing/hnsw.index")
        faiss.write_index(self.index, index_file_path)
        return

    def load_index(self):
        """
        Load index from index file
        Returns HNSW index
        """
        base_dir = os.path.dirname(__file__)
        index_file_path = os.path.join(base_dir, f"../data/{self.dataset}/indexing/hnsw.index")
        self.index = faiss.read_index(index_file_path)
        return