import numpy as np
import faiss

# Define the dimension of the vectors
d = 64

# Define the size of the database
nb = 100000

# Define the number of queries
nq = 10000

# Set the random seed for reproducibility
np.random.seed(1234)

# Generate random vectors for the database
xb = np.random.random((nb, d)).astype('float32')
print(xb[0])

# Add a unique identifier to each vector in the database
xb[:, 0] += np.arange(nb) / 1000.
print(len(xb[0]))

# Generate random vectors for the queries
xq = np.random.random((nq, d)).astype('float32')

# Add a unique identifier to each vector in the queries
xq[:, 0] += np.arange(nq) / 1000.

# Create an HNSW index
index = faiss.IndexHNSWFlat(d, 32)

print("before add")
# Add the vectors to the index
index.add(xb)
print("after add")

# Perform a search
D, I = index.search(xq, 4)

# Print the results
print(D[:5])  # Neighbors of the first 5 queries
print(D[-5:])  # Neighbors of the last 5 queries
print(I[:5])  # Neighbors of the first 5 queries
print(I[-5:])  # Neighbors of the last 5 queries

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# size = os.path.getsize("data/airline_reviews/embeddings/bge.parquet")
# print(size/1000000)