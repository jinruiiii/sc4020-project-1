import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.embedder import Embedder
from utils.load_data import load_csv
from alive_progress import alive_bar

# just change input_csv_file, output_csv_file and column_name to embed different dataset

def get_file_paths():
    base_dir = os.path.dirname(__file__)
    input_csv_file = os.path.join(base_dir, "../data/starbucks/raw_data/starbucks.csv")
    output_embedding_file = os.path.join(base_dir, "../data/airline_reviews/embeddings/bge.parquet")
    return input_csv_file, output_embedding_file


def embed_description(description, embedder, prog_bar: alive_bar):
    prog_bar()
    return embedder.embed(description).squeeze(0).numpy()

def process_embeddings(data, column_name, embedder):
    with alive_bar(data.shape[0]) as bar:
        data['embedding'] = data[column_name].map(lambda desc: embed_description(desc, embedder, bar))
    return data

def save_embeddings(data, output_filepath):
    # Extract only the embedding column and save
    embeddings_df = pd.DataFrame(data['embedding'].tolist())
    embeddings_df.to_parquet(output_filepath)

def main():
    # Get file paths
    input_csv_file, output_embedding_file = get_file_paths()

    # Load data
    data = load_csv(input_csv_file)

    # Initialize embedder
    embedder = Embedder()

    # Process embeddings
    column_name = "Review"
    data = process_embeddings(data, column_name, embedder)
    
    # Save the embeddings to a file
    save_embeddings(data, output_embedding_file)
    return

if __name__ == "__main__":
    main()