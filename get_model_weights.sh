#!/usr/bin/env bash

mkdir -p model_weights/metaclip
mkdir -p model_weights/evaclip
mkdir -p model_weights/mobileclip

wget https://huggingface.co/BAAI/EVA-CLIP-18B/resolve/main/EVA_CLIP_18B_psz14_s6B.fp16.pt -P model_weights/evaclip
wget https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt -P model_weights/metaclip
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt -P model_weights/mobileclip
