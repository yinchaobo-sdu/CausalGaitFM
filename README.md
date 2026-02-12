# CausalGaitFM

A Causal Representation-Driven Cross-Domain Foundation Model for Clinical Gait Analysis.

## Overview

CausalGaitFM achieves robust cross-domain gait analysis through four integrated mechanisms:

1. **Structural Causal Model (SCM)**: Disentangles domain-invariant causal factors from domain-specific confounders
2. **Counterfactual Data Augmentation**: Synthesizes diverse training scenarios via domain intervention
3. **Invariant Risk Minimization (IRM)**: Learns representations predictive across all training domains
4. **Mamba State-Space Backbone**: Efficiently captures long-range temporal dependencies with linear complexity

The model jointly predicts **fall risk**, **frailty level**, and **disease classification** through a multi-task learning framework with uncertainty weighting.

## Project Structure

```
CausalGaitFM/
├── README.md
├── pyproject.toml
├── CausalGaitFM.pdf              # Paper
├── project/
│   ├── model.py                  # CausalGaitModel + GaitDecoder
│   ├── train.py                  # Training script (CLI)
│   ├── run_experiments.py        # Experiment runner (ablation, baselines, cross-domain)
│   ├── requirements.txt
│   ├── models/
│   │   ├── backbone.py           # Multi-scale Mamba TemporalEncoder
│   │   ├── scm.py                # SCM_Layer with DAG learning
│   │   ├── heads.py              # MultiTaskHeads + OrdinalHead (frailty)
│   │   └── causal_gait.py        # Simplified integration scaffold
│   ├── baselines/
│   │   ├── erm.py                # ERM baseline
│   │   ├── dann.py               # DANN baseline (gradient reversal)
│   │   ├── coral.py              # Deep CORAL baseline
│   │   └── irm_baseline.py       # IRM standalone baseline
│   ├── data/
│   │   ├── download.py           # Dataset download scripts
│   │   ├── preprocess.py         # Per-dataset preprocessing pipelines
│   │   └── dataset.py            # GaitDataset, DataLoader, eval protocols
│   └── utils/
│       ├── losses.py             # IRM penalty, HSIC loss
│       ├── metrics.py            # Accuracy, Macro-F1, MAE
│       ├── augment.py            # Counterfactual augmentation
│       └── visualization.py      # t-SNE latent space plots
└── outputs/                      # Training outputs (auto-created)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/CausalGaitFM.git
cd CausalGaitFM

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r project/requirements.txt

# Optional: install Mamba SSM for GPU-accelerated SSM
pip install mamba-ssm>=2.2
```

## Local RTX 4060 One-Command Pipeline (Windows)

The repository now includes an end-to-end pipeline entrypoint:

```bash
python -m project.pipeline --profile local_4060_full --device cuda --auto-batch
```

Recommended local setup (single 4060):

```bash
py -3 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.10.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv\Scripts\python -m project.pipeline --profile local_4060_full --device cuda --auto-batch
```

Pipeline steps:
1. environment check
2. dataset download
3. dataset preprocess
4. smoke training
5. full experiments
6. benchmark
7. paper tables/figures generation

Expected outputs:
- `outputs/cross_domain/results.json`
- `outputs/ablation/results.json`
- `outputs/baselines/results.json`
- `outputs/in_domain/results.json`
- `outputs/single_task/results.json`
- `outputs/benchmark/benchmark_results.json`
- `outputs/paper/all_tables.md`
- `outputs/paper/figure2a_ablation.png`
- `outputs/paper/figure2d_efficiency.png`

Typical runtime on a single RTX 4060 (fallback backend): around 12-30 hours for full pipeline.

## One-Click Start / Stop / Resume (Windows PowerShell)

Start training in background:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_train.ps1 -RunId r4060_train
```

Graceful stop (writes `outputs/control/<run_id>.stop`, waits for safe checkpoint save):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_run.ps1 -RunId r4060_train
```

Resume training from `outputs/<run_id>/last_model.pth`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/resume_train.ps1 -RunId r4060_train
```

Start full pipeline in background:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_pipeline.ps1 -RunId r4060_full
```

Resume pipeline from `outputs/<run_id>/pipeline_state.json`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/resume_pipeline.ps1 -RunId r4060_full
```

Watch pipeline progress in real time (stage + substage + ETA):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/watch_pipeline.ps1 -RunId r4060_full
```

Watch progress with recent log tail:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/watch_pipeline.ps1 -RunId r4060_full -ShowLog -LogLines 30
```

`pipeline_state.json` now includes:

- `progress_percent`
- `stages_total` / `stages_completed`
- `current_stage`
- `current_substage`
- `eta_seconds`
- `started_at`
- `last_error`

Default optimization profile applied by scripts:

- `--device cuda`
- `--use-amp true --amp-dtype fp16`
- `--allow-tf32 true --cudnn-benchmark true`
- `--num-workers 4 --pin-memory true --persistent-workers true --prefetch-factor 4`
- `--auto-batch --min-batch-size 4`
- stop/resume controls: `control_dir=outputs/control`, `check_stop_every=20`, `save_every_steps=200`

## Quick Start

### 1. Download and Preprocess Data

```bash
# Download all 6 benchmark datasets
python -m project.data.download

# Preprocess into unified format
python -m project.data.preprocess
```

### 2. Train CausalGaitFM

```bash
# Cross-domain evaluation (train on 5 domains, test on held-out domain)
python -m project.train --eval-mode cross_domain --target-domain daphnet

# In-domain 5-fold cross-validation
python -m project.train --eval-mode in_domain --fold 0

# Leave-one-subject-out evaluation
python -m project.train --eval-mode loso --fold 0

# Quick test with dummy data
python -m project.train --use-dummy-data --num-epochs 5
```

### 3. Run Experiments

```bash
# Full cross-domain evaluation across all target domains (Table 2)
python -m project.run_experiments --experiment cross_domain

# Ablation study (Figure 2)
python -m project.run_experiments --experiment ablation

# Baseline comparisons (Table 2)
python -m project.run_experiments --experiment baselines

# In-domain 5-fold CV (Table 3)
python -m project.run_experiments --experiment in_domain

# Run all experiments
python -m project.run_experiments --experiment all

# Use dummy data for debugging
python -m project.run_experiments --experiment ablation --use-dummy-data --num-epochs 5
```

## Datasets

Six publicly available gait/activity recognition datasets:

| Dataset | Subjects | Samples | Sensors | Classes | Focus |
|---------|----------|---------|---------|---------|-------|
| Daphnet | 10 | 23,614 | 3 IMU | 2 | Freezing of Gait |
| UCI-HAR | 30 | 10,299 | Smartphone | 6 | Activity Recognition |
| PAMAP2 | 9 | 376,417 | 3 IMU | 12 | Activity Monitoring |
| MHEALTH | 10 | 161,280 | Body-worn | 12 | Health Monitoring |
| WISDM | 29 | 1,098,207 | Smartphone | 6 | Activity Recognition |
| Opportunity | 4 | 701,366 | 72 sensors | 17 | Context Recognition |

## Model Architecture

```
Input x [B, T, D]
    |
    v
TemporalEncoder (Multi-scale Mamba SSM, bidirectional)
    |
    v
SCM_Layer (Structural Causal Model)
    |-- z_c: Causal factors (32-dim, DAG structure)
    |-- z_d: Domain factors (16-dim, Gaussian latent)
    |
    +---> MultiTaskHeads (disease, fall risk, frailty ordinal)
    +---> GaitDecoder (reconstruction + counterfactual generation)
```

## Training Objective (Eq. 11)

```
L = L_recon + beta1*L_KL + beta2*L_cls + beta3*L_IRM + beta4*L_DAG + beta5*L_HSIC + beta6*L_MT + beta7*L_CF
```

| Component | Description | Weight |
|-----------|-------------|--------|
| L_recon | Reconstruction MSE | 1.0 |
| L_KL | Domain KL divergence | 0.01 |
| L_cls | Classification cross-entropy | 1.0 |
| L_IRM | IRM penalty (annealed) | dynamic |
| L_DAG | DAG acyclicity constraint | 0.01 |
| L_HSIC | Causal-domain independence | 0.01 |
| L_MT | Multi-task uncertainty loss | 1.0 |
| L_CF | Counterfactual classification | 0.5 |

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Mamba d_model | 128 |
| Mamba d_state | 16 |
| Mamba layers | 4 |
| Temporal scales | (1, 2, 4) |
| Causal dim (d_c) | 32 |
| Domain dim (d_s) | 16 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Epochs | 100 (early stopping) |
| IRM warmup | 500 iterations |

## Citation

```bibtex
@article{causalgaitfm2026,
  title={CausalGaitFM: A Causal Representation-Driven Cross-Domain Foundation Model for Clinical Gait Analysis},
  author={Anonymous},
  year={2026}
}
```

## License

MIT License
