import json
import os
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def consensus_scoring(all_candidates: List[str]) -> List[float]:
    """
    Compute the Conensus score (CIDEr score) for a list of candidate captions using the consensus scoring method.

    Parameters:
    - all_candidates (List[str]): A list of candidate captions.

    Returns:
    - List[float]: The Conensus score (CIDEr score) scores for the candidate captions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_matrix = vectorizer.fit_transform(all_candidates)
    tf_idf_tensor = torch.tensor(tf_idf_matrix.todense(), dtype=torch.float64).to(device)
    similarity_scores = torch.mm(tf_idf_tensor, tf_idf_tensor.transpose(0, 1))
    concensus_scores = (
        ((similarity_scores.sum(dim=-1) - 1) / (similarity_scores.size(1) - 1)).cpu().detach().numpy().tolist()
    )

    return concensus_scores


def rulebase_filter(caption_list: List[str]):
    """
    Filter captions based on rule-based criteria.

    Parameters:
    - caption_list (List[str]): List of captions to filter.
    """
    filtered_captions = [
        caption
        for caption in caption_list
        if len(caption.split()) > 5 and caption.count(",") < 3 and caption.count(".") < 2
    ]
    return filtered_captions if filtered_captions else caption_list


def itm_filter(scores_dict: Dict[str, List[str]]):
    """
    Filter captions based on ITM scores.

    Parameters:
    - scores_dict (Dict[str, List[str]]): Dictionary with keys "captions" and "scores".
    """
    captions = scores_dict["captions"]
    scores = scores_dict["scores"]

    # Get indexes of 50% of the captions with the highest scores
    top_50_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: len(scores) // 2]
    top_50_captions = [captions[i] for i in top_50_idx]
    return top_50_captions


if __name__ == "__main__":
    # Load Configs
    cfg = OmegaConf.load("configs.yaml")

    print("Loading blip2 ITM scores...")
    with open("/workspace/data/scores/blip2_itm_scores.json", "r") as f:
        blip2_itm_scores = json.load(f)
    print("Loaded blip2 ITM scores.")

    result_dict = {}
    for file in tqdm(blip2_itm_scores.keys()):
        filtered_captions = itm_filter(blip2_itm_scores[file])
        filtered_captions = rulebase_filter(filtered_captions)
        consensus_score = consensus_scoring(filtered_captions)

        result_dict[file] = {"captions": filtered_captions, "scores": consensus_score}
    score_file_path = os.path.join(cfg.DIR.Score, "itm_filtered_consensus.json")
    with open(score_file_path, "w") as f:
        json.dump(result_dict, f)
    print("Scoring completed. Saving scores to", score_file_path)
