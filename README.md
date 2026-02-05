# DrugR - Optimizing Molecular Drugs with LLM-based Explicit Reasoning

Cite this: DOI: 00.0000/xxxxxxxxxx

DrugR is an LLM-based framework for molecular optimization that introduces explicit, step-by-step pharmacological reasoning into the generation process. It integrates domain-specific continual pretraining, supervised fine-tuning via reverse data engineering, and self-balanced multi-granular reinforcement learning to improve key ADMET properties while preserving structural similarity and target binding affinity.

## Highlights

- Explicit, interpretable reasoning traces for each optimization step.
- Multi-stage training pipeline tailored for molecular tasks.
- Evaluation utilities for OOD and property-focused assessments.
- Open-source scripts for data processing, training, and generation.

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

- Split and validate chemistry datasets:

```bash
python src/split_chem_data.py --help
python src/check_train_val_overlap.py --help
```

- Build reverse data engineering inputs:

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
