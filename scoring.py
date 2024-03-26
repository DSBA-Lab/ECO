import argparse
import os


import open_clip
import pandas as pd
import torch
from omegaconf import OmegaConf

import torch
import json
from PIL import Image
from tqdm import tqdm
import os

@torch.no_grad()
def clip_score(model, tokenizer, preprocess, captions, cfg):
    """
    Calculate CLIP scores for given images and captions.

    Parameters:
    - model: CLIP model for encoding images and text.
    - tokenizer: Tokenizer for processing captions.
    - preprocess: Preprocessing function for images.
    - captions (DataFrame): DataFrame where columns are filenames and values are captions.
    - cfg: Configuration object with attributes for directories and score model name.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dict = {}
    for file in tqdm(captions.columns):
        image_path = os.path.join(cfg.DIR.Origin, f"images_20k/{file}")
        with Image.open(image_path).convert("RGB") as img:
            image = preprocess(img).unsqueeze(0).to(device)

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        caption_list = captions[file].dropna().tolist()
        tokenized_caption_list = tokenizer(caption_list).to(device)
        caption_features = model.encode_text(tokenized_caption_list)
        caption_features /= caption_features.norm(dim=-1, keepdim=True)

        clip_scores = (image_features @ caption_features.T).detach().cpu().tolist()[0]
        result_dict[file] = {"captions": caption_list, "scores": clip_scores}

    score_file_path = os.path.join(cfg.DIR.Score, f"{cfg.score_model}_scores_1.json")
    print("Scoring completed. Saving scores to", score_file_path)
    with open(score_file_path, "w") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    # args
    valid_models = ["mobileclip", "openclip", "evaclip", "metaclip", "blip_itc", "blip_itm"]
    parser = argparse.ArgumentParser(description="Generate CLIP scores for the candidate captions.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=valid_models,
        help="The model to use for scoring the captions. Choose from: " + ", ".join(valid_models),
    )
    args = parser.parse_args()

    # Load Configs
    cfg = OmegaConf.load("configs.yaml")
    cfg.score_model = args.model

    # Load Captions
    print("Loading captions...")
    captions = pd.read_csv(os.path.join(cfg.DIR.Origin, "candidate_captions.csv"), encoding="cp1252").T
    captions.columns = captions.iloc[0]
    captions = captions.drop(captions.index[0])
    captions.reset_index(drop=True, inplace=True)
    print("Captions loaded.")

    # Torch Setting
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    print("Loading model...")
    if args.model == "evaclip":
        from evaclip import create_model_and_transforms, get_tokenizer

        model, _, preprocess = create_model_and_transforms(
            "EVA-CLIP-18B",
            pretrained=os.path.join(cfg.DIR.Weight, "evaclip", "EVA_CLIP_18B_psz14_s6B.fp16.pt"),
            force_custom_clip=True,
            device=device,
        )
        tokenizer = get_tokenizer("EVA-CLIP-18B")
    elif args.model == "metaclip":
        from meta_clip import create_model_and_transforms
        from open_clip import SimpleTokenizer

        model, _, preprocess = create_model_and_transforms(
            "ViT-bigG14-quickgelu",
            pretrained=os.path.join(cfg.DIR.Weight, "metaclip", "G14_fullcc2.5b.pt"),
            device=device,
        )
        tokenizer = SimpleTokenizer()
    elif args.model == "mobileclip":
        from mobileclip import create_model_and_transforms, get_tokenizer

        model, _, preprocess = create_model_and_transforms(
            "mobileclip_b",
            pretrained=os.path.join(cfg.DIR.Weight, "mobileclip", "mobileclip_blt.pt"),
            device=device,
        )
        tokenizer = get_tokenizer("mobileclip_b")
    elif args.model == "openclip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-bigG-14")

    print(f"Model {args.model} loaded.")
    print("Scoring captions...")
    if args.model in valid_models[:4]:
        clip_score(model, tokenizer, preprocess, captions, cfg)
