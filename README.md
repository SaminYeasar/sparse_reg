# Sparse-reg: Improving Sample Complexity in Offline Reinforcement Learning using Sparsity


## Setup
```
conda create -n sparse_reg python=3.9
conda activate sparse_reg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
```
---
## Run baseline
`python main.py --env_name halfcheetah-expert-v2 --algo_name IQL --buffer_size 10000`


## Run sparse 
`python main.py --env_name halfcheetah-expert-v2 --algo_name IQL --buffer_size 10000 --activate_sparse_reg --keep_ratio 0.05`
* `activate_sparse_reg`: activates sparse network
* `keep_ratio`: determines fraction of active param in scale 1. Ex: For $95\%$ sparse network `--keep_ratio 0.05`
* `turn_off_sparse_reg_at`: controls iterative update of sparse-reg, default=200k steps. For single-shot set value to 0.

---
## Citation:
```
@misc{arnob2025sparseregimprovingsamplecomplexity,
      title={Sparse-Reg: Improving Sample Complexity in Offline Reinforcement Learning using Sparsity}, 
      author={Samin Yeasar Arnob and Scott Fujimoto and Doina Precup},
      year={2025},
      eprint={2506.17155},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.17155}, 
}
```