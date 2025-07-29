# HyperAim: Hypergraph Contrastive Learning with Adaptive Multi-frequency Filters
## Requirements
- python==3.9.21
- numpy==1.23.0
- scikit-learn==1.3.2
- torch==1.12.1+cu116
- torch-geometric==1.7.2
- pandas==2.0.1
- scipy==1.9.1
- tqdm==4.67.1

## How to Run HyperAim
- python main.py --dataset [DATASET NAME] --task [TASK NAME] --device [DEVICE NAME] e.g., python main.py --dataset cora_cocitation --task finetune --device cuda:0 
