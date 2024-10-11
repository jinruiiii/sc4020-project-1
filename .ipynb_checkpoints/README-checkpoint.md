# SC4020 Project 1

## Setup

For faiss setup:

```
python==3.9
numpy==1.26.0
faiss-cpu==1.7.4
torch==2.1.1
transformers==4.44.2
pyarrow
```

might need to use conda to install faiss if doesnt work

```
conda install -c pytorch faiss-cpu
```

Other troubleshooting:
[zsh: segmentation fault when running faiss on CPU](https://github.com/facebookresearch/faiss/issues/2099)

## About FAISS

### Parameters to set

- M: no. of neighbours during insertion
- efConstruction: no. of nearest neighbours to explore during construction (optional)
  - must be set at construction of index
- efSearch: no. of nearest neighbours to explore during search (optional)
  - can be set at or after construction of index
- mL: normalization factor (optional)

For more information: [Read about HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/)
