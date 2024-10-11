import pandas as pd
import os
import faiss


def load_csv(file_path):
    return pd.read_csv(file_path)


def load_parquet(file_path):
    return pd.read_parquet(file_path)


def load_index(index_file_path):
    return faiss.read_index(index_file_path)
