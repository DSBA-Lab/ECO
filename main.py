import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import score_ensemble, standardize_scores


def final_scoring(scores_dict: Dict[str, Dict[str, float]], cfg: Any) -> None:
    """
    Calculate final scores using consensus and ensemble scores, and update predictions.

    Parameters:
    - scores_dict (Dict[str, Dict[str, Dict[str, float]]): Dictionary of scores.
    - cfg (OmegaConf): The configuration file.
    """
    cons_score_dict = scores_dict["consensus"]
    clip_score_dict = {k: v for k, v in scores_dict.items() if k != "consensus"}
    ens_score_dict = score_ensemble(*list(clip_score_dict.values()))
    pred_file_path = os.path.join(cfg.DIR.Origin, "pred.csv")
    pred = pd.read_csv(pred_file_path)
    final_scores_dict = {}

    for file in tqdm(cons_score_dict):
        caption_list = list(cons_score_dict[file])
        cons_scores = np.array(list(cons_score_dict[file].values()))
        ens_scores = np.array([ens_score_dict[file][caption] for caption in caption_list])

        total_scores = cfg.WEIGHT.Consensus_Score * cons_scores + cfg.WEIGHT.CLIP_Score * ens_scores
        final_scores_dict[file] = {"captions": caption_list, "scores": total_scores.tolist()}

        max_idxs = np.argsort(total_scores)[-2:][::-1]
        max_caps = [caption_list[idx] for idx in max_idxs]

        if abs(total_scores[max_idxs[0]] - total_scores[max_idxs[1]]) < cfg.THRESH.Short_Threshold:
            max_caption = min(max_caps, key=lambda x: len(x.split()))
        else:
            max_caption = max_caps[0]

        pred.loc[pred["filename"] == file, "caption"] = max_caption

    mmdd = datetime.now().strftime("%m%d")
    ens_list_str = "_".join(model_name for model_name in list(scores_dict.keys()))
    # ens_list_str = "_".join([next((k for k, v in globals().items() if v == ens_item), None) for ens_item in ens_list])
    filename = f"{mmdd}_pred_{cfg.WEIGHT.Consensus_Score}_{cfg.WEIGHT.CLIP_Score}_{ens_list_str}_{cfg.THRESH.Short_Threshold}.csv"
    print(f"Saving predictions to {filename}")
    pred.to_csv(os.path.join(cfg.DIR.Result, filename), index=False)


def load_and_standardize_scores(file_name, score_directory):
    """
    Attempts to load and standardize scores from a specified JSON file.
    :param file_name: The name of the JSON file containing the scores.
    :param score_directory: The directory where the score files are located.
    :return: The standardized scores if successful, None otherwise.
    """
    try:
        return standardize_scores(os.path.join(score_directory, file_name))
    except Exception as e:
        print(f"Error loading {file_name}. Please check the file path.")
        print(f"Error details: {e}")
        exit(1)


if __name__ == "__main__":
    cfg = OmegaConf.load("configs.yaml")

    scores_dict = {}
    score_models = [
        ("consensus", "itm_filtered_consensus.json", True),
        ("mobileclip", "mobileclip_scores.json", cfg.MODEL.MobileCLIP),
        ("openclip", "openclip_scores.json", cfg.MODEL.OpenCLIP),
        ("evaclip", "evaclip_scores.json", cfg.MODEL.EvaCLIP),
        ("metaclip", "metaclip_scores.json", cfg.MODEL.MetaCLIP),
        ("blipitc", "blip_itc_scores.json", cfg.MODEL.BlipITC),
    ]

    for model_name, file_name, condition in score_models:
        if condition:
            scores = load_and_standardize_scores(file_name, cfg.DIR.Score)
            if scores is not None:
                scores_dict[model_name] = scores

    final_scoring(scores_dict, cfg)
