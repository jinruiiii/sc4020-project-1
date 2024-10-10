import faiss
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_data import load_parquet, load_csv, load_index
from utils.embedder import Embedder
import time

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
                # # initialize vectorizer
                # self.vectorizer = Embedder()

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

        Returns the created HNSW Index and time taken to create the index in nanoseconds
        """
        start = time.time_ns()
        if efConstruction is not None:
            self.efConstruction = efConstruction
            self.index.hnsw.efConstruction = efConstruction

        if M is not None and mL is not None:
            self.index.set_default_probas(M, mL)

        self.index.add(self.data_store_embeddings)  # build the index
        
        # change efSearch after adding data
        if efSearch is not None:
            self.efSearch = efSearch
            self.index.hnsw.efSearch = efSearch
        end = time.time_ns()
        duration = end-start
        return self.index, duration
    
    def save_index(self):
        """
        Saves built HNSW index into index file
        """
        base_dir = os.path.dirname(__file__)
        index_file_path = os.path.join(base_dir, f"../data/{self.dataset}/indexing/hnsw_m={self.M}_c={self.efConstruction}_s={self.efSearch}.index")
        faiss.write_index(self.index, index_file_path)
        return

def main():
    dataset = "airline_reviews"
    m_vals = [2**i for i in range(4, 10)]
    con_vals = [2**i for i in range(10)]
    search_vals = [2**i for i in range(10)]

    # note that efSearch should not make a difference in time
    index_dict = {
        "m": [],
        "efConstruction": [],
        "efSearch": [],
        "time_taken": []
    }

    for m in m_vals:
        for c in con_vals:
            for s in search_vals:
                hnsw = HNSW(dataset, "bge", m)
                hnsw.index, duration = hnsw.construct_graph(efConstruction=c, efSearch=s)
                hnsw.save_index()
                index_dict["m"].append(m)
                index_dict["efConstruction"].append(c)
                index_dict["efSearch"].append(s)
                index_dict["time_taken"].append(duration)
    
    df = pd.DataFrame.from_dict(index_dict)
    base_dir = os.path.dirname(__file__)
    output_file = os.path.join(base_dir, f"../data/{dataset}/indexing/hnsw_construction.parquet")
    df.to_parquet(output_file)

if __name__ == "__main__":
    main()