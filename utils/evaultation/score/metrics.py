import json
import math

from utils.load_data import load_parquet
from typing import List


def remove_query_from_predictions(documents, query):
    if query in documents:
        index = documents.index(query)
        documents.pop(index)
    else:
        documents.pop()
    return documents


def get_mean_precision_k(
        ground_truth_path: str,
        file_paths: List[str],
        k: int,
        num_relevant_docs: int
) -> List[float]:
    """
    Function to get mean precision
    :param ground_truth_path: Path to the ground truth parquet file
    :param file_paths: List[str], of data paths to add to comparisons
    :param k:
    :param num_relevant_docs:
    :return: List[float] corresponding to the mean precision of the files in `file_paths`
    """

    ground_truth = load_parquet(ground_truth_path)
    num_questions = ground_truth.shape[0]
    data_dfs = [load_parquet(p) for p in file_paths]

    mean_precision_scores = []

    for df in data_dfs:
        precision_score = 0
        for i in range(num_questions):
            question = ground_truth["question"][i]
            set1 = set(ground_truth["top_k"][i][1:num_relevant_docs+1])
            set2 = set(remove_query_from_predictions(list(json.loads(df["top_k"][i])["Review"].values()), question)[0:k])
            true_positive = len(set1.intersection(set2))
            precision_score += true_positive/k
        mean_precision_scores.append(precision_score/num_questions)
    return mean_precision_scores


def get_mean_reciprocal_rank(
        ground_truth_path: str,
        file_paths: List[str],
        k: int
) -> List[float]:
    """
    Function to get MAP
    :param ground_truth_path: Path to the ground truth parquet file
    :param file_paths: List[str], of data paths to add to comparisons
    :param k: Number of results to compare
    :return: List[float] corresponding to the MAP of the files in `file_paths`
    """

    ground_truth = load_parquet(ground_truth_path)
    num_questions = ground_truth.shape[0]
    data_dfs = [load_parquet(p) for p in file_paths]

    mean_reciprocal_ranks = []

    def get_reciprocal_rank(_most_relevant_document, _predictions):
        try:
            rank = _predictions.index(_most_relevant_document) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    for df in data_dfs:
        reciprocal_rank = 0
        for i in range(num_questions):
            question = ground_truth["question"][i]
            most_relevant_document = ground_truth["top_k"][i][1]
            predictions = remove_query_from_predictions(list(json.loads(df["top_k"][i])["Review"].values()), question)[0:k]
            reciprocal_rank += get_reciprocal_rank(most_relevant_document, predictions)
        mean_reciprocal_ranks.append(reciprocal_rank/num_questions)
    return mean_reciprocal_ranks


def get_mean_ndcg_k(
        ground_truth_path: str,
        file_paths: List[str],
        k: int
):
    """
    Function to get nDCG
    :param ground_truth_path: Path to the ground truth parquet file
    :param file_paths: List[str], of data paths to add to comparisons
    :param k: Number of results to compare
    :return: List[float] corresponding to the nDCG of the files in `file_paths`
    """

    ground_truth = load_parquet(ground_truth_path)
    num_questions = ground_truth.shape[0]
    data_dfs = [load_parquet(p) for p in file_paths]

    mean_ndcgs = []

    def compute_relevance_scores(_ground_truth, predictions):
        relevance_map = {doc: len(_ground_truth) - score for score, doc in enumerate(_ground_truth)}
        scores = [relevance_map.get(doc, 0) for doc in predictions]
        return scores

    for df in data_dfs:
        ndcg = 0
        for i in range(num_questions):
            question = ground_truth["question"][i]
            relevance_score = compute_relevance_scores(ground_truth["top_k"][i][1:k+1], remove_query_from_predictions(list(json.loads(df["top_k"][i])["Review"].values()), question)[0:k])
            true_relevance = list(range(k,0,-1))
            idcg = 0
            dcg = 0
            for j in range(len(relevance_score)):
                # formula
                idcg += true_relevance[j] / math.log(j+2,2)
                dcg += relevance_score[j] / math.log(j+2,2)
            ndcg += dcg/idcg
        mean_ndcgs.append(ndcg/num_questions)
    return mean_ndcgs
