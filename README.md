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

## 📦 Code and Checkpoints

We open-source the following resources to support reproducibility and future research:

- Training and inference code
- Model checkpoints
- Data preprocessing and evaluation scripts

Repository: TBD

## Project Structure

- `src/`: training, generation, evaluation, and data-prep scripts
- `data/`: datasets and intermediate artifacts (local paths)
- `figures/`: paper figures and visual assets
- `eval_output/`, `output/`, `result/`: experiment outputs
- `ChemDFM/`, `ExLLM/`, `ModelCenter/`: external components or integrations
- `test/`: evaluation and sanity tests

## Quick Start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements0517.txt
```

If you use the scripts under `src/`, install extra dependencies if needed:

```bash
pip install -r src/requirements.txt
```

### 2) Data Preparation

Split and validate chemistry datasets:

```bash
python src/split_chem_data.py --help
python src/check_train_val_overlap.py --help
```

Build reverse data engineering inputs:

```bash
python src/prepare_reverse_engineering_input.py --help
```

See `src/REVERSE_DATA_ENGINEERING_README.md` and `src/HOW_TO_USE_PAPER_MATERIALS.md` for details.

### 3) Training

Common training entrypoints:

```bash
bash src/chem_split_training.sh
bash src/mixed_chem_training.sh
bash src/propellant_improved.sh
```

Guidance for addressing overfitting and selecting hyperparameters:

- `chem_training_guide.md`

### 4) Generation

```bash
python src/generate.py --help
python src/generate_ood.py --help
```

### 5) Evaluation

```bash
python src/eval_chem_model.py --help
bash src/evaluate_existing_checkpoint.sh
```

Binding energy evaluation notes:

- `src/BINDING_ENERGY_EVALUATION_README.md`

## Paper Materials

This repository includes draft paper materials and latex sources:

- `explicit_reasoning_dataset.tex`
- `model_and_training_settings.tex`
- `src/TRAINING_PIPELINE_PAPER.md`
- `src/REVERSE_DATA_ENGINEERING_PAPER.md`

## Reproducibility Notes

- Use the provided scripts for consistent data splitting and validation.
- Keep training and validation datasets strictly separated.
- Adjust learning rate and steps as recommended in `chem_training_guide.md`.

## Citation

```bibtex
@article{DrugR,
  title = {DrugR: Optimizing Molecular Drugs with LLM-based Explicit Reasoning},
  author = {Liu, Haoran and Full Name and Zeng, Zheni},
  journal = {TBD},
  year = {2025},
  doi = {00.0000/xxxxxxxxxx}
}
```

## License

See `LICENSE`.

## Contact

For questions or collaboration, please open an issue or contact the authors.
