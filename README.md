# NICE2024
![pipeline](docs/pipeline.png)
Official repository of Team DSBA for the NICE 2024 Challenge



**Table of Contents**
- [Setup](#setup)
  - [Environment](#environment)
  - [Data](#data)
  - [Model Weight](#model-weight)
- [How to run](#run)
  - [Score Generation](#score-generation)
- [Reference](#reference)
## Setup
### Environment
This guide provides detailed instructions for setting up the necessary environments to run the code for different models including `BLIP2,` `EvaCLIP`, `MobileCLIP`, `MetaCLIP`, and `OpenCLIP`. Each model requires a unique environment for optimal performance and compatibility.

You can install the required dependencies by running the following commands:

```bash
# BLIP2
conda create -n blip2 python=3.8
conda activate blip2
pip install salesforce-lavis omegaconf
```
```bash
# EVA-CLIP
conda create -n evaclip python=3.10
conda activate evaclip
git clone https://github.com/baaivision/EVA.git
cd EVA/EVA-CLIP-18B
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install omegaconf
```
```bash
# CLIP Env (MobileCLIP, MetaCLIP, OpenCLIP)
conda create -n clipenv python=3.10
conda activate clipenv
# Install Mobile CLIP
git clone https://github.com/apple/ml-mobileclip.git
cd ml-mobileclip
pip install -e .
pip install omegaconf
```
or you can install environment from dockerfile by running the following command:
```bash
docker build -t nice2024 .
docker run -it --gpus all nice2024
```

### Data
The `original data` can be downloaded from the [competition website](https://codalab.lisn.upsaclay.fr/competitions/16930#participate) or you can make use of the `get_data.sh` script to download the data.
The `scores` can be downloaded from the following links: [Google Drive](https://drive.google.com/drive/folders/1-p2ps4DWpexhSQj4IP6pMPMgcS4KthM-?usp=sharing)

  
  ```bash
  ├── data
│   ├── original_data
│   │   ├── candidate_captions.csv
│   │   ├── images_20k
│   │   └── pred.csv
│   ├── results
│   │   └── #Result csv file will be saved here
│   └── scores
│       ├── blip_itc_scores.json # You can generate by 1. Score Generation
│       ├── blip_itm_scores.json # You can generate by 1. Score Generation
│       ├── evaclip_scores.json # You can generate by 1. Score Generation
│       ├── itm_filtered_consensus.json # You can generate by 2. Consensus Score Generation
│       ├── metaclip_scores.json # You can generate by 1. Score Generation
│       ├── mobileclip_scores.json # You can generate by 1. Score Generation
│       └── openclip_scores.json # You can generate by 1. Score Generation
  ```

### Model Weight
The model weights can be downloaded from the following links:
Weights of openclip and blip2 is automatically downloaded when you run the score generation script.

<div align="center">

|    `model_name`     | `model weight` |
|:-------------------:|:--------------:|
| [EvaCLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B) | [EVA_18B_psz14.fp16](https://huggingface.co/BAAI/EVA-CLIP-18B/resolve/main/EVA_CLIP_18B_psz14_s6B.fp16.pt) (`36.7GB`) |
| [MetaCLIP](https://github.com/facebookresearch/MetaCLIP)  | [ViT-bigG-14-quickgelu](https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt) (`28.38GB`) |
| [MobileCLIP](https://github.com/apple/ml-mobileclip/tree/main) | [mobileclip_blt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt) (`571.46MB`) |

</div>

or you can download all the model weights by running the following command:
```bash
source ./scripts/00_get_model_weights.sh   # Files will be downloaded to `model_weights` directory.
```
  ```bash
├── model_weights
│   ├── evaclip
│   │   └── EVA_CLIP_18B_psz14_s6B.fp16.pt
│   ├── metaclip
│   │   └── G14_fullcc2.5b.pt
│   └── mobileclip
│       └── mobileclip_blt.pt
  ```

## How to run
### 1. CLIP Score Generation
You need to prepare score files for each vision-language model to generate the final submission file. 
You can either generate the scores for each model or you can download the scores from the following links: [Google Drive](https://drive.google.com/drive/folders/1-p2ps4DWpexhSQj4IP6pMPMgcS4KthM-?usp=sharing) \
You need a 80GB of VRAM for running the EVA-CLIP 18B Model.

```bash
source ./scripts/01_evaclip_score.sh #script_filename [evaclip_score.sh, metaclip_score.sh, mobileclip_score.sh, openclip_score.sh, blip_itc_score.sh, blip_itm_score.sh]
```
or you can download all the scores by running the following command:
```bash
source ./scripts/01_all_score.sh
```
### 2. Consensus Score Generation
After finishing generating scores for each VL models. Now you can generate consensus scores by running the following command:
```bash
source ./scripts/02_consensus_score.sh
```
### 3. Ensemble Score and Submission File Generation
At last, fuse all the scores and generate the final submission file by running the following command:
```bash
source ./scripts/03_ensemble_score.sh
```



## Reference
### BLIP2
```bibtex
@inproceedings{li2023blip,
  title={Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models},
  author={Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven},
  booktitle={International conference on machine learning},
  pages={19730--19742},
  year={2023},
  organization={PMLR}
}
```
### EvaCLIP
```bibtex
@article{EVA-CLIP-18B,
  title={EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters}, 
  author={Quan Sun and Jinsheng Wang and Qiying Yu and Yufeng Cui and Fan Zhang and Xiaosong Zhang and Xinlong Wang},
  journal={arXiv preprint arXiv:2402.04252},
  year={2023}
}
```
### MetaCLIP
```bibtex
@inproceedings{xu2023metaclip,
   title={Demystifying CLIP Data},
   author={Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer and Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:2309.16671},
   year={2023}
}
```
### MobileCLIP
```bibtex
@InProceedings{mobileclip2024,
  author = {Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel},
  title = {MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024},
}
```
### OpenCLIP
```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

## Acknowledgements
Our codebase is built using multiple opensource contributions. We would like to thank the authors of the following repositories for their valuable contributions