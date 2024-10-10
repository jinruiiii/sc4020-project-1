import faiss
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_data import load_parquet

def get_input_file_path(dataset):
    base_dir = os.path.dirname(__file__)
    input_embedding_file = os.path.join(base_dir, f"../data/{dataset}/embeddings/bge.parquet")
    return input_embedding_file

def get_output_file_path(dataset, nbits):
    base_dir = os.path.dirname(__file__)
    output_indexing_file = os.path.join(base_dir, f"../data/{dataset}/indexing/lsh{nbits}.index")
    return output_indexing_file

def build_index(embeddings, dims, nbits):
    # Create and configure FAISS LSH index
    index = faiss.IndexLSH(dims, nbits)
    index.add(embeddings)
    return index

def save_index(index, index_file_path):
    faiss.write_index(index, index_file_path)

def main():
    # Get file paths
    input_embedding_file = get_input_file_path("airline_reviews")

    embeddings = np.vstack(load_parquet(input_embedding_file).values)
    dims = embeddings.shape[1]
    for nbits in [2,4,8,16,32,64,128,256,512,768]:

        # Build LSH index
        index = build_index(embeddings, dims, nbits)

        # Save LSH index
        output_indexing_file = get_output_file_path("airline_reviews", nbits)
        save_index(index, output_indexing_file)




if __name__ == "__main__":
    main()
