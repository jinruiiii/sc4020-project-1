import faiss
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from utils.load_data import load_parquet

def get_file_paths():
    base_dir = os.path.dirname(__file__)
    input_embedding_file = os.path.join(base_dir, "../data/airline_reviews/embeddings/bge.parquet")
    output_indexing_file = os.path.join(base_dir, "../data/airline_reviews/indexing/lsh.index")
    return input_embedding_file, output_indexing_file

def build_index(embeddings, dims, nbits):
    # Create and configure FAISS LSH index
    index = faiss.IndexLSH(dims, nbits)
    index.add(embeddings)
    return index

def save_index(index, index_file_path):
    faiss.write_index(index, index_file_path)

def main():
    # Get file paths
    input_embedding_file, output_indexing_file = get_file_paths()

    embeddings = np.vstack(load_parquet(input_embedding_file)["embedding"].values)
    dims = embeddings.shape[1]
    nbits=64

    # Build LSH index
    index = build_index(embeddings, dims, nbits)

    # Save LSH index
    save_index(index, output_indexing_file)




if __name__ == "__main__":
    main()
