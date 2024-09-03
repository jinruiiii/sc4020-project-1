import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.VSMAlgo import VSM
from scripts.create_hnsw_index import HNSW

print("initializing")
hnsw_index = HNSW(dataset="starbucks", embedding_type="bge", M=16)

print("constructing")
index = hnsw_index.construct_graph(efConstruction= 100, efSearch= 50)
print("index", index)

print("get embeddings")
data = hnsw_index.get_embeddings()

print("start searching")
results = hnsw_index.search(query= data[:10], k=1)
print(results)