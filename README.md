# RegSimAgg: Regularized Similarity-based Aggregation for Federated Learning

A complete, reproduction-oriented implementation of **RegSimAgg** (*Khan et al., 2022*) and the FeTS 2022 Challenge modules ported to **Flower 1.x**, **PyTorch**, and **MONAI**.

Paper: *"Regularized Weight Aggregation in Networked Federated Learning for Glioblastoma Segmentation"* (arXiv:2301.12617).  
Reference GitHub: [`dskhanirfan/FeTS2022`](https://github.com/dskhanirfan/FeTS2022).

---

## Implemented Modules

### 1. Collaborator Selection (`regsimagg/collaborator_selector.py`)
- **`custom_percentage_collaborator_selector_without_repetition`**:
  - Implements the 20% sliding window over shuffled collaborators.
  - Ensures equal participation across all institutions over cycles.
  - Guarantees that selected collaborator combinations are never repeated in future rounds.
- **`equal_partitions`**: Pads collaborator arrays if not divisible by window size.
- **`custom_percentage_collaborator_selector`**: Standard sliding window selector.
- **`all_collaborators_train`**: Baseline selecting all collaborators every round.
- **`one_collaborator_on_odd_rounds`**: Alternating selector choosing the fastest client on odd rounds.
- **`CollaboratorSelector`**: Flower-native stateful wrapper for ServerApp integration.

### 2. Hyperparameter Scheduling (`regsimagg/hyperparameters.py`)
- **`constant_hyper_parameters`**: Default FeTS challenge setting (`lr=5e-5`, `epochs=1.0`).
- **`train_less_each_round`**: Epochs decay by $0.9^{\min(r, 10)}$ for initial rounds.
- **`fixed_number_of_batches`**: Trains for a flat count of 16 batches, cycling over local data if needed.

### 3. Aggregation Engine (`regsimagg/regsimagg.py`)
- **`my_sum`**: Exact multi-dimensional recursive summation from the FeTS GitHub codebase.
- **`similarity_weights`**:
  - `github_compat`: $d_c = |\mathrm{sum}(\hat{p}) - \mathrm{sum}(p_c)|$ using the recursive `my_sum` function from the author's public repository.
  - `paper_l1`: $d_c = \|p_c - \hat{p}\|_1 = \sum |p_c - \hat{p}|$ matching Eq. (2) from the paper.
  - Normalization $u_c = s_c / \sum s_j$ (Eq. 3).
- **`sample_weights`**: $v_c = N_c / \sum N_i$ (Eq. 4).
- **`base_weights`**: $w_c = (u_c + v_c) / \sum(u_i + v_i)$ (Eq. 5).
- **`regularize_weights`**: Post-round-10 parameter shift regularization (Eq. 6) damping erratic updates.
- **`reg_sim_weighted_average_layerwise`**: Layer-by-layer callback reproduction matching Intel OpenFL's execution model.
- **Challenge Baselines**:
  - `fedavg_aggregate` (Standard FedAvg)
  - `clipped_aggregate` (80th percentile delta clipping)
  - `fedavgm_aggregate` (Server Momentum $\beta=0.9$, Server Learning Rate $\eta=1.0$)

### 4. MONAI 3D Model, Loss & Metrics (`regsimagg/model.py`)
- **3D U-Net Architecture**: Powered by `monai.networks.nets.UNet` with instance normalization and residual units.
- **BraTS / FeTS Subregion Mappings**:
  - **Whole Tumor (WT)**: Labels 1, 2, 4
  - **Tumor Core (TC)**: Labels 1, 4
  - **Enhancing Tumor (ET)**: Label 4
- **Loss Function**: MONAI multi-label `DiceLoss(sigmoid=True)`.
- **Evaluation**: Computes Dice similarity and Hausdorff95 distance for WT, TC, and ET.

### 5. Data Pipeline & Partitions (`regsimagg/data.py`, `regsimagg/partitioner.py`)
- Multi-modal NIfTI loader (4 channels: T1, T1ce, T2, FLAIR).
- Non-zero brain intensity normalization and spatial cropping/padding.
- Built-in synthetic 4-channel NIfTI dataset generator for immediate offline testing.
- FeTS CSV partition parser (`partitioning_1.csv`, `partitioning_2.csv`) with symlink support.

### 6. Leaderboard Inference (`regsimagg/inference.py`)
- **`model_outputs_to_disc`**: Runs sliding window inference over validation volumes and exports multi-class NIfTI predictions with original affine headers.

---

## Quick Start

### 1. Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\Activate.ps1

# Install package and dependencies
pip install -e .
```

### 2. Run All Unit Tests

Verify all 14 unit tests across algorithms, model forward passes, metrics, and data loaders:

```bash
pytest -v tests/
```

### 3. Run Standalone Experiment Simulation

Run a multi-client experiment simulation (automatically generates synthetic 4D NIfTI data if real data is not yet downloaded):

```bash
python run_experiment.py --rounds 5 --clients 5 --fraction 0.4 --method regsimagg
```

### 4. Run with Flower Simulation Engine

```bash
flwr run . --run-config "num-server-rounds=5 fraction-train=0.2"
```

For the paper's full challenge setting:
```bash
flwr run . --run-config "num-server-rounds=500 fraction-train=0.2 local-epochs=1 learning-rate=0.00005 regularization-round=10"
```

### 5. Generate Challenge NIfTI Outputs for Evaluation

```python
from regsimagg.inference import model_outputs_to_disc

model_outputs_to_disc(
    data_path="/path/to/validation_data",
    output_path="./challenge_model_outputs",
    native_model_path="./experiment_outputs/best_model.pt",
)
```
