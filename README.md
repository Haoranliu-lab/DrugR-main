# DrugR: Optimizing Molecular Drugs through LLM-based Explicit Reasoning


[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://arxiv.org/pdf/2602.08213.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2602.08213-b31b1b)](https://arxiv.org/abs/2602.08213)


### Authors

Haoran Liu<sup>1</sup>,  Zheni Zeng<sup>2*</sup>,  Yukun Yan<sup>3</sup>,  Yuxuan Chen<sup>4</sup>,  Yunduo Xiao<sup>5</sup>

<sup>1</sup> School of Biological Science and Medical Engineering,  Beihang Advanced Innovation Center for Biomedical Engineering,  Beihang University, Beijing 100191,China.  <sup>2</sup> Nanjing University.  <sup>3</sup> Tsinghua University.  
<sup>4</sup> School of Electronic and Computer Engineering, Peking University, Shenzhen, China.
<sup>5</sup> School of Computer Science and Engineering,  South China University of Technology, Guangzhou, China.



## 📌 Overview

DrugR is a large language model (LLM)-based framework for molecular drug optimization that introduces explicit, step-by-step pharmacological reasoning into the optimization process. Unlike prior implicit or black-box approaches, DrugR generates interpretable rationales for each molecular modification step, enabling reliable multi-objective optimization of drug-like molecules. DrugR focuses on improving key ADMET properties while preserving structural similarity and target-binding affinity, advancing toward automated and knowledge-driven drug discovery.

## 🧠 Motivation

Molecular optimization is a fundamental yet challenging task in drug discovery due to:

- Multi-objective complexity
- Poor generalization across drug types
- Lack of interpretability in decision-making

Although large language models exhibit strong reasoning capabilities, general-domain LLMs lack molecular expertise, while domain-specific models often suffer from limited reasoning ability. DrugR bridges this gap by combining domain knowledge with explicit reasoning-driven optimization.

## 🚀 Key Contributions

- Explicit Pharmacological Reasoning: step-by-step reasoning chains for optimization.
- LLM-based Optimization Framework: joint optimization of multiple properties.
- Domain-Specific Continual Pretraining: injects molecular and drug knowledge.
- Reverse Data Engineering for SFT: high-quality instruction-reasoning-output triples.
- Self-Balanced Multi-Granular RL: multi-property optimization with similarity control.

## 🧪 Task Definition

Given:

- An original drug molecule
- Corresponding pharmacological properties

DrugR generates:

- An optimized candidate molecule (SMILES)
- An explicit reasoning chain, including the targeted property objectives, structural modification rationale, key functional group preservation analysis, and justification for the expected ADMET improvements


Subject to:

- Improvement in targeted ADMET properties
- Structural fingerprint similarity >= 0.6
- Preservation of functional consistency with the original molecule

Current experiments focus on three small-molecule drug categories:

- Anti-inflammatory
- Antihypertensive
- Hypoglycemic drugs

## 🏗️ Framework Architecture

DrugR consists of three main stages:

1. Domain Continual Pretraining: injects molecular structure and pharmacology knowledge.
2. Supervised Fine-Tuning (SFT): reverse-engineered optimization trajectories with explicit reasoning.
3. Reinforcement Learning Optimization: self-balanced, multi-granular rewards for multiple objectives.

This design enables comprehensive enhancement across multiple drug properties while maintaining molecular validity and similarity.

## 📊 Experimental Results

Experimental evaluations demonstrate that:

- DrugR consistently improves multiple ADMET properties.
- Structural similarity and target-binding affinity are preserved.
- Explicit reasoning chains provide clear and actionable optimization rationales.

These results highlight DrugR's effectiveness and interpretability compared to implicit or non-reasoning-based baselines.

## 🔍 Interpretability and Reasoning

A key advantage of DrugR is its explicit reasoning process, which:

- Explains why each molecular modification is made
- Links structural changes to pharmacological outcomes
- Enables human-in-the-loop analysis and validation

This makes DrugR suitable for scientific discovery workflows, not just black-box optimization.

## 📦 Code and Resources

We open-source the following resources to support reproducibility and future research:

- Model architecture and training framework
- Simulation and evaluation modules
- Data processing pipelines
- Inference scripts and utilities

The repository includes organized modules under:

- `models/` — model implementation
- `simulator/` — property evaluation and simulation components
- `data/` — processed datasets and supporting files


## 🗂️ Project Structure

- `data/`  
  Training data, reasoning datasets, and evaluation inputs.

- `models/`  
  Core implementation of DrugR, including:
  - `swift_pretrain.py` — domain continual pretraining
  - `swift_sft.py` — supervised fine-tuning with explicit reasoning
  - `grpo_vllm.py` — reinforcement learning optimization
  - `general_dataset.py` — dataset processing utilities
  - `llm_test.py` — model inference testing

- `simulator/`  
  Evaluation and simulation modules:
  - `admet_evaluater.py` — ADMET property evaluation
  - `docking_evaluater.py` — molecular docking evaluation
  - `lms_judge.py` — language model reasoning evaluation
  - `reasoning_richness_evaluator.py` — reasoning quality assessment

- `requirements.txt`  
  Dependency specification



## 🚀 Quick Start (Advanced)

This section provides an end-to-end workflow covering environment setup, sanity check, three-stage training, and evaluation.

```bash
# 1) Clone
git clone  https://github.com/Haoranliu-lab/DrugR-main.git
cd DrugR-main

# 2) Environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 3) Sanity check (make sure the code can run)
python -c "import torch; print('torch:', torch.__version__)"
python models/llm_test.py --help || true

# -------------------------
# Stage 1: Domain Continual Pretraining
# -------------------------
python models/swift_pretrain.py --help
# Example (replace with your real args):
# python models/swift_pretrain.py \
#   --data_path data/reasoning.jsonl \
#   --output_dir checkpoints/pretrain

# -------------------------
# Stage 2: Supervised Fine-Tuning (SFT)
# -------------------------
python models/swift_sft.py --help
# Example:
# python models/swift_sft.py \
#   --train_path data/reasoning.jsonl \
#   --output_dir checkpoints/sft

# -------------------------
# Stage 3: Reinforcement Learning Optimization (GRPO)
# -------------------------
python models/grpo_vllm.py --help
# Example:
# python models/grpo_vllm.py \
#   --policy_dir checkpoints/sft \
#   --output_dir checkpoints/rl

# -------------------------
# Evaluation
# -------------------------

# ADMET evaluation
python simulator/admet_evaluater.py --help
# Example:
# python simulator/admet_evaluater.py --smiles "CCO"

# Docking evaluation
python simulator/docking_evaluater.py --help
# Example:
# python simulator/docking_evaluater.py --smiles "CCO" --target YOUR_TARGET

# Reasoning quality evaluation
python simulator/reasoning_richness_evaluator.py --help
# Example:
# python simulator/reasoning_richness_evaluator.py --input data/open_question.json
```

---

### 🧪 Example I/O (Reasoning + Molecule + Evaluation)

Below is an illustrative example of how DrugR outputs an optimized molecule together with an explicit reasoning chain and evaluation signals.

**Input**
- Drug type: `anti-inflammatory`
- Original molecule (SMILES): `CC(=O)OC1=CC=CC=C1C(=O)O`
- Targeted ADMET objectives: improve solubility, reduce toxicity

**Output (example format)**
```
Optimized SMILES:
  CC(=O)OC1=CC=C(O)C=C1C(=O)O

Similarity (fingerprint):
  0.71  (>= 0.60 ✔)

Predicted ADMET changes:
  Solubility:   +0.18
  Toxicity:     -0.12
  Permeability: +0.05

Docking (optional):
  Binding affinity: preserved / improved

Reasoning Chain:
1. Added a hydroxyl substitution to enhance polarity and improve solubility.
2. Preserved the core aromatic scaffold and carboxyl group to maintain functional consistency.
3. Avoided high-risk substructures associated with toxicity while keeping similarity above threshold.
```

**Notes**
- The above output is a recommended interface format for readability and reproducibility.
- If you are using your own evaluation pipeline, ensure the printed metrics at least include:
  similarity, targeted ADMET changes, and the reasoning chain.

## Citation

```bibtex
@article{liu2026DrugR,
  title  = {DrugR: Optimizing Molecular Drugs through LLM‐based Explicit Reasoning},
  author = {Haoran Liu and Zheni Zeng and Yukun Yan and Yuxuan Chen and Yunduo Xiao},
  journal = {arXiv preprint arXiv:2602.08213},
  year   = {2026},
  url    = {https://arxiv.org/abs/2602.08213}
}


## License

See `LICENSE`.

## Contact

For questions or collaboration, please open an issue or contact the authors.
