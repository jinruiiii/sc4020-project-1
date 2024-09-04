import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.create_hnsw_index import HNSW

print("initializing")
hnsw_index = HNSW(dataset="starbucks", embedding_type="bge", M=32)

print("constructing")
index = hnsw_index.construct_graph()
print("index", index)

hnsw_index.save_index()

print("get embeddings")
data = hnsw_index.data_store_embeddings

print("start searching")
# indices, results = hnsw_index.search(queries= ["4 Hours before takeoff we received a Mail stating a cryptic message that there are disruptions to be expected as there is a limit on how many planes can leave at the same time. So did the capacity of the Heathrow Airport really hit British Airways by surprise, 4h before departure? Anyhow - we took the one hour delay so what - but then we have been forced to check in our Hand luggage. I travel only with hand luggage to avoid waiting for the ultra slow processing of the checked in luggage. Overall 2h later at home than planed, with really no reason, just due to incompetent people. Service level far worse then Ryanair and triple the price. Really never again. Thanks for nothing."], k=3)
indices, results = hnsw_index.search(queries= ["Amber and LaDonna at the Starbucks on Southwest Parkway are always so warm and welcoming. There is always a smile in their voice when they greet you at the drive-thru. And their customer service is always spot-on, they always get my order right and with a smile. I would actually give them more than 5 stars if they were available."], k=3)
print(results)

print("load index")
hnsw_index.load_index()
indices, results = hnsw_index.search(queries= ["Amber and LaDonna at the Starbucks on Southwest Parkway are always so warm and welcoming. There is always a smile in their voice when they greet you at the drive-thru. And their customer service is always spot-on, they always get my order right and with a smile. I would actually give them more than 5 stars if they were available."], k=3)
print("loaded")