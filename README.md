# CS-ADS Replication

This repository contains the replication code for **CS-ADS: Activation Decomposition and Steering for LLM Backdoor Remediation**.

The codebase has two entry points:

- `main.py` — benchmark replication for safety and utility.
- `csads.py` — testing CS-ADS on customized samples. Examples are included in this file. 

---

# 1. Code Structure

```text
.
├── csads.py
├── data_util.py
├── eval.py
├── intervener.py
├── main.py
└── util.py
```

- `main.py` — experiment driver and command-line interface.
- `csads.py` — customized-sample inference.
- `eval.py` — safety, utility, and AutoPoison evaluation functions.
- `intervener.py` — CS-ADS activation collection, decomposition, and steering.
- `util.py` — activation decomposition utilities.
- `data_util.py` — dataset loading, triggers/toxins, poisoned-data preparation, and ASR evaluation.

---

# 2. Environment

The replication environment uses:

```text
Python       3.10
PyTorch      2.1.0
Transformers 4.48.2
lm-eval      0.4.7
```

Example installation:

```bash
conda create -n csads python=3.10
conda activate csads

pip install torch==2.1.0
pip install transformers==4.48.2
pip install lm-eval==0.4.7
pip install peft pandas tqdm
```

A CUDA-capable GPU is recommended.

---

# 3. Evaluation Settings

The evaluated methods are:

```text
CS-ADS
No-Defense
Prepending
Random Noise
```

The CS-ADS intervention strength follows:

```python
effective_strength = strength * scaler
```

where:

- `strength` is the model-specific base value.
- `scaler` is the command-line multiplier.

Default model groups are configured in `main.py`:

```python
all_typical_models = [
    ("./Llama-2-7b-HSI-Backdoored", None, 0.75),
    ("./Llama-2-7b-CI-Backdoored", None, 0.75),
]

all_autopoison_models = [
    ("TaiGary/AutoPoison", None, 0.75),
]

all_utility_models = (
    all_typical_models
    + all_autopoison_models
)
```

Safety keeps **typical HSI/CI** and **AutoPoison** separate. Utility evaluates the merged model list.

---

# 4. Benchmark Replication with `main.py`

Show all options:

```bash
python main.py --help
```

Main options:

```text
--run
    safety
    utility
    all
    asr

--safety-set
    typical
    autopoison
    all

--task
    arc_challenge
    arc_easy
    mmlu
    all

--layer-study
    auto
    typical
    front
    mid
    all
    none

--scaler
    float

--autopoison-strength
    float
```

## Safety

Typical HSI/CI safety:

```bash
python main.py \
    --run safety \
    --safety-set typical
```

AutoPoison safety:

```bash
python main.py \
    --run safety \
    --safety-set autopoison
```

All safety settings:

```bash
python main.py \
    --run safety \
    --safety-set all
```

No-defense ASR only:

```bash
python main.py --run asr
```

Typical safety reports:

```text
Model
Method
Effective Strength
Triggered ASR
No-Trigger ASR
```

AutoPoison safety reports:

```text
Model
Method
Effective Strength
ASR
```

## Utility

Supported tasks:

```text
arc_challenge
arc_easy
mmlu
```

Run one task:

```bash
python main.py \
    --run utility \
    --task arc_challenge
```

Run all utility tasks:

```bash
python main.py \
    --run utility \
    --task all
```

Utility is evaluated on:

```python
all_typical_models + all_autopoison_models
```

and reports:

```text
Model
Task
CS-ADS Strength
CS-ADS
No-Defense
Prepending
Random Noise
```

Run the complete replication:

```bash
python main.py \
    --run all \
    --safety-set all \
    --task all \
    --layer-study auto \
    --scaler 1.0
```

---

# 5. Customized Samples with `csads.py`

Use `csads.py` for direct testing on customized inputs instead of the benchmark pipeline:

```bash
python csads.py
```

Typical uses include:

- testing a triggered prompt;
- testing a benign/no-trigger prompt;
- comparing defended and undefended outputs;
- testing a customized instruction;
- checking a model checkpoint before full benchmark evaluation.

Use `main.py` for benchmark replication and `csads.py` for customized inference.

---

# 6. References and Citation

## Implementation Credit

Our intervention module is inspired by:

https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb

## Typical Backdoor Settings — TUBA

```bibtex
@inproceedings{he-etal-2025-tuba,
    title = "{TUBA}: Cross-Lingual Transferability of Backdoor Attacks in {LLM}s with Instruction Tuning",
    author = "He, Xuanli  and
      Wang, Jun  and
      Xu, Qiongkai  and
      Minervini, Pasquale  and
      Stenetorp, Pontus  and
      Rubinstein, Benjamin I. P.  and
      Cohn, Trevor",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.848/",
    doi = "10.18653/v1/2025.findings-acl.848",
    pages = "16504--16544",
    ISBN = "979-8-89176-256-5"
}
```

## AutoPoison

```bibtex
@inproceedings{NEURIPS2023_c2a8060f,
    author = {Shu, Manli and Wang, Jiongxiao and Zhu, Chen and Geiping, Jonas and Xiao, Chaowei and Goldstein, Tom},
    booktitle = {Advances in Neural Information Processing Systems},
    doi = {10.52202/075280-2703},
    editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
    pages = {61836--61856},
    publisher = {Curran Associates, Inc.},
    title = {On the Exploitability of Instruction Tuning},
    url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/c2a8060fd22744b38177d9e428a052e0-Paper-Conference.pdf},
    volume = {36},
    year = {2023}
}
```

## CS-ADS

If you use this code or CS-ADS in your research, please cite:

```bibtex
@inproceedings{zhong-etal-2026-activation,
    title = "Activation Decomposition and Steering for {LLM} Backdoor Remediation",
    author = "Zhong, Lingfeng  and
      Xu, Qiongkai  and
      Naseem, Usman",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.2025/",
    doi = "10.18653/v1/2026.acl-long.2025",
    pages = "43713--43737",
    ISBN = "979-8-89176-390-6"
}
```
