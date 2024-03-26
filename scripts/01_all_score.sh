#!/usr/bin/env bash
conda activate blip2
echo "Calculating BLIP-ITC scores..."
python scoring.py --model blip_itm
echo "Calculating BLIP-ITM scores..."
python scoring.py --model blip_itc
echo "Calculating evaclip CLIP scores..."
conda deactivate
conda activate evaclip
python scoring.py --model evaclip
echo "Calculating metaclip CLIP scores..."
conda deactivate
conda activate clipenv
python scoring.py --model metaclip
echo "Calculating openclip CLIP scores..."
python scoring.py --model openclip
echo "Calculating mobileclip CLIP scores..."
python scoring.py --model mobileclip
