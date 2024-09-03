from enum import Enum


class AlgoType(str, Enum):
    VSM = "vsm"
    HNSW = "hnsw"
    LSH = "lsh"
