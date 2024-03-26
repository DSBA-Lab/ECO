import json
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def consensus_scoring(all_candidates: List[str], device: str = "cuda") -> np.ndarray:
    """
    Compute the CIDEr score for a list of candidate captions using the consensus scoring method.

    Parameters:
    - all_candidates (List[str]): A list of candidate captions.
    - device (str): The device to use for computation (default: 'cuda').

    Returns:
    - np.ndarray: The CIDEr scores for the candidate captions.
    """
    # Ensure PyTorch is using the specified device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Use TfidfVectorizer to vectorize the captions
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_matrix = vectorizer.fit_transform(all_candidates)

    # Convert the TF-IDF matrix to a dense PyTorch tensor and transfer to the device
    tf_idf_tensor = torch.tensor(tf_idf_matrix.todense(), dtype=torch.float64).to(device)

    # Compute cosine similarity
    similarity_scores = torch.mm(tf_idf_tensor, tf_idf_tensor.transpose(0, 1))

    # Calculate the average cosine similarity as the CIDEr score
    cider_scores = ((similarity_scores.sum(dim=-1) - 1) / (similarity_scores.size(1) - 1)).cpu().detach().numpy()

    return cider_scores


def standardize(score: Any, mean: float, std_dev: float) -> float:
    """
    Standardize a score using the mean and standard deviation.

    Parameters:
    - score (Any): The score to standardize.
    - mean (float): The mean of the scores.
    - std_dev (float): The standard deviation of the scores.

    Returns:
    - float: The standardized score.
    """
    return (score - mean) / std_dev


def standardize_scores(score_filepath: str) -> Dict[str, Dict[str, float]]:
    """
    Open score file and standardize the scores.

    Parameters:
    - score_filepath (str): The path to the score file.

    Returns:
    - std_scores (Dict[str, Dict[str, float]]): The standardized scores.
    """
    with open(score_filepath, "r") as f:
        score_data = json.load(f)

    scores = {
        file: {caption: score_data[file]["scores"][idx] for idx, caption in enumerate(score_data[file]["captions"])}
        for file in (score_data.keys())
    }

    score_val = []
    for key in scores.keys():
        score_val.extend(list(scores[key].values()))
    score_mean, score_std = np.mean(score_val), np.std(score_val)

    std_scores = {
        file: {caption: standardize(scores[file][caption], score_mean, score_std) for caption in (scores[file].keys())}
        for file in (scores.keys())
    }

    return std_scores


def score_ensemble(*score_dicts: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Ensemble the multiple CLIP scores into a single score.

    Parameters:
    - score_dicts: Multiple score dictionaries.

    Returns:
    - Dict[str, Dict[str, float]]: The standardized ensemble scores.
    """
    ensemble_dict = {}
    for file in tqdm(next(iter(score_dicts)).keys()):
        ensemble_score = np.zeros(len(next(iter(score_dicts))[file].values()))
        for score_dict in score_dicts:
            ensemble_score += np.array(list(score_dict[file].values()))
        ensemble_dict[file] = {caption: ensemble_score[idx] for idx, caption in enumerate(score_dicts[0][file].keys())}

    ensemble_values = [score for scores in ensemble_dict.values() for score in scores.values()]
    ensemble_mean, ensemble_std = np.mean(ensemble_values), np.std(ensemble_values)

    std_ensemble_dict = {
        file: {
            caption: standardize(score, ensemble_mean, ensemble_std) for caption, score in ensemble_dict[file].items()
        }
        for file in ensemble_dict
    }
    return std_ensemble_dict
