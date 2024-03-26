import argparse
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm


@torch.no_grad()
def clip_score(model, tokenizer, preprocess, captions, device, cfg):
    """
    Calculate CLIP scores for given images and captions.

    Parameters:
    - model: CLIP model for encoding images and text.
    - tokenizer: Tokenizer for processing captions.
    - preprocess: Preprocessing function for images.
    - captions (DataFrame): DataFrame where columns are filenames and values are captions.
    - device: Device to run the model on.
    - cfg: Configuration object with attributes for directories and score model name.

    """
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


@torch.no_grad()
def itc_score(model, image_embedding, text_ids, text_atts):
    """
    Calculate ITC scores for given images and captions.

    Parameters:
    - model: BLIP model
    - image_embedding: Image embeddings
    - text_ids: Tokenized text
    - text_atts: Attention mask for text_ids
    """
    device = image_embedding.device
    image_embedding = image_embedding.repeat(text_ids.shape[0], 1, 1).to(device)
    image_atts = torch.ones(image_embedding.size()[:-1], dtype=torch.long).to(device)

    query_tokens = model.query_tokens.expand(image_embedding.shape[0], -1, -1)
    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embedding,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    image_feats = F.normalize(model.vision_proj(query_output.last_hidden_state), dim=-1)

    text_output = model.Qformer.bert(
        text_ids,
        attention_mask=text_atts,
        return_dict=True,
    )
    text_feat = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

    sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
    sim, _ = torch.max(sims, dim=1)

    return sim.cpu().squeeze().numpy().tolist()


@torch.no_grad()
def itm_score(model, image_embedding, text_ids, text_atts):
    """
    Calculate ITM scores for given images and captions.

    Parameters:
    - model: BLIP model
    - image_embedding: Image embeddings
    - text_ids: Tokenized text
    - text_atts: Attention mask for text_ids
    """
    device = image_embedding.device
    image_embedding = image_embedding.repeat(text_ids.shape[0], 1, 1).to(device)
    image_atts = torch.ones(image_embedding.size()[:-1], dtype=torch.long).to(device)

    query_tokens = model.query_tokens.expand(image_embedding.shape[0], -1, -1)
    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
    attention_mask = torch.cat([query_atts, text_atts], dim=1)

    output_itm = model.Qformer.bert(
        text_ids,
        query_embeds=query_tokens,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embedding,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
    itm_logit = model.itm_head(vl_embeddings).mean(dim=1)
    itm_logit = torch.nn.functional.softmax(itm_logit, dim=-1)[:, 1]
    return itm_logit.cpu().numpy().tolist()


@torch.no_grad()
def blip_score(model, vis_processors, captions, device, cfg):
    """
    Calculate BLIP (ITC, ITM) scores for given images and captions.
    Parameters:
    - model: BLIP model
    - vis_processors: Image processors for BLIP model
    - captions (DataFrame): DataFrame where columns are filenames and values are captions.
    - device: Device to run the model on.
    - cfg: Configuration object with attributes for directories and score model name.
    """
    result_dict = {}
    for file in tqdm(captions.columns):
        image = (
            vis_processors["eval"](Image.open(os.path.join(cfg.DIR.Origin, f"images_20k/{file}")).convert("RGB"))
            .unsqueeze(0)
            .to(device)
        )
        imgae_embedding = model.ln_vision(model.visual_encoder(image)).float()

        caption_list = captions[file].dropna().values.tolist()
        text = model.tokenizer(caption_list, truncation=True, padding=True, max_length=32, return_tensors="pt").to(
            device
        )
        if cfg.score_model[-3:] == "itc":
            score = itc_score(model, imgae_embedding, text.input_ids, text.attention_mask)
        elif cfg.score_model[-3:] == "itm":
            score = itm_score(model, imgae_embedding, text.input_ids, text.attention_mask)
        result_dict[file] = {"captions": caption_list, "scores": score}

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
    captions = pd.read_csv(os.path.join(cfg.DIR.Origin, "candidate_captions.csv"), encoding="ISO-8859-1").T
    captions.columns = captions.iloc[0]
    captions = captions.drop(captions.index[0])
    captions.reset_index(drop=True, inplace=True)
    print("Captions loaded.")

    # Device Setting
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    if args.model == "evaclip":
        from eva_clip import create_model_and_transforms, get_tokenizer

        model, _, preprocess = create_model_and_transforms(
            "EVA-CLIP-18B",
            pretrained=os.path.join(cfg.DIR.Weight, "evaclip", "EVA_CLIP_18B_psz14_s6B.fp16.pt"),
            force_custom_clip=True,
            device=device,
        )
        tokenizer = get_tokenizer("EVA-CLIP-18B")
    elif args.model == "metaclip":
        from open_clip import SimpleTokenizer

        from meta_clip import create_model_and_transforms

        model, _, preprocess = create_model_and_transforms(
            "ViT-bigG-14-quickgelu",
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
    elif args.model[:4] == "blip":
        from lavis.models import load_model_and_preprocess

        model, vis_processors, text_processors = load_model_and_preprocess(
            "blip2_image_text_matching", "coco", device=device, is_eval=True
        )

    print(f"Model {args.model} loaded.")
    print("Scoring captions...")
    if args.model in valid_models[:4]:  # CLIP Scores
        clip_score(model, tokenizer, preprocess, captions, device, cfg)
    else:  # BLIP_ITC, BLIP_ITM Scores
        blip_score(model, vis_processors, captions, device, cfg)
