"""Experiment runner replicating FeTS `run_challenge_experiment`.

Allows running RegSimAgg simulations either standalone or via Flower,
and logs per-round performance metrics for reproduction comparison.
Directly supports official FeTS 2022 dataset directories and partition CSVs.
"""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import time
from typing import Optional
import pandas as pd
import torch

from regsimagg.collaborator_selector import CollaboratorSelector
from regsimagg.hyperparameters import get_hyperparameters_for_round
from regsimagg.regsimagg import (
    base_weights,
    regularize_weights,
    aggregate_states,
    state_to_numpy,
)
from regsimagg.model import get_model, evaluate_subregions
from regsimagg.data import make_loader, generate_synthetic_fets_data
from regsimagg.inference import model_outputs_to_disc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_challenge_experiment(
    aggregation_method: str = "regsimagg",
    collaborator_selector: str = "without_repetition",
    hyperparameters_schedule: str = "constant",
    distance_mode: str = "github_compat",
    regularization_round: int = 10,
    rounds_to_train: int = 10,
    num_clients: int = 5,
    fraction_train: float = 0.2,
    learning_rate: float = 5e-5,
    data_root: str | Path = "./data",
    train_data_dir: Optional[str | Path] = None,
    val_data_dir: Optional[str | Path] = None,
    partition_csv: Optional[str | Path] = None,
    device: str = "cpu",
    output_dir: str | Path = "./experiment_outputs",
    model_type: str = "monai_unet",
    epochs: Optional[float] = None,
    batches: Optional[int] = None,
    predict_val: bool = False,
    val_limit: Optional[int] = None,
    patch_size: int = 64,
    loss_type: str = "dice",
) -> pd.DataFrame:
    """Simulates the FeTS challenge experiment loop matching the paper's specification."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dev.type == "cpu":
        num_cpus = os.cpu_count() or 4
        torch.set_num_threads(num_cpus)
        torch.set_num_interop_threads(max(1, num_cpus // 4))
        logger.info("Enabled CPU acceleration: %d threads, %d interop threads", num_cpus, max(1, num_cpus // 4))

    # Resolve training data source
    use_real_fets = False
    if train_data_dir is not None and partition_csv is not None:
        train_path = Path(train_data_dir)
        csv_path = Path(partition_csv)
        if train_path.exists() and csv_path.exists():
            use_real_fets = True
            df_part = pd.read_csv(csv_path)
            part_col = [c for c in df_part.columns if "partition" in c.lower()][0]
            unique_partitions = sorted(df_part[part_col].unique())
            all_collaborators = [str(pid) for pid in unique_partitions]
            num_clients = len(all_collaborators)
            logger.info("Using real FeTS dataset from %s with %s (%d institutions)",
                        train_path, csv_path.name, num_clients)
        else:
            logger.warning("Specified train_data_dir or partition_csv does not exist. Falling back to data_root.")

    if not use_real_fets:
        data_path = Path(data_root)
        if not data_path.exists() or not list(data_path.glob("client_*")):
            logger.info("No client data found at %s. Generating synthetic BraTS data...", data_path)
            generate_synthetic_fets_data(data_path, num_clients=num_clients, samples_per_client=2)
        all_collaborators = [f"client_{i}" for i in range(num_clients)]

    # Initialize global model
    global_model = get_model(model_type=model_type).to(dev)
    previous_global_state = None

    selector = CollaboratorSelector(method=collaborator_selector, percentage=fraction_train)

    round_records = []
    try:
        from monai.losses import DiceLoss, DiceCELoss
        if loss_type == "dice_ce":
            loss_fn = DiceCELoss(sigmoid=True)
        else:
            loss_fn = DiceLoss(sigmoid=True)
    except ImportError:
        from regsimagg.client_app import MultiLabelDiceLoss
        loss_fn = MultiLabelDiceLoss()

    p_size = (patch_size, patch_size, patch_size)

    logger.info("Starting experiment: Aggregation=%s, Selector=%s, Distance=%s, Loss=%s, PatchSize=%s, Rounds=%d, Clients=%d",
                aggregation_method, collaborator_selector, distance_mode, loss_type, p_size, rounds_to_train, num_clients)

    if batches is not None and batches > 0 and hyperparameters_schedule == "constant":
        hyperparameters_schedule = "fixed_batches"

    start_time = time.time()

    for r in range(rounds_to_train):
        round_start = time.time()

        # 1. Choose training collaborators
        selected_cols = selector.select(all_collaborators, fl_round=r)
        logger.info("--- Round %d: Selected %s ---", r, selected_cols)

        # 2. Get hyperparameters for this round
        lr, round_epochs, round_batches = get_hyperparameters_for_round(
            schedule=hyperparameters_schedule,
            fl_round=r,
            learning_rate=learning_rate,
            epochs_per_round=epochs,
            batches_per_round=batches,
        )

        # 3. Local training on selected clients
        client_states = []
        client_num_examples = []

        for col in selected_cols:
            if use_real_fets:
                loader = make_loader(
                    data_root=train_path,
                    partition_csv=csv_path,
                    partition_id=col,
                    batch_size=1,
                    patch_size=p_size,
                    is_train=True,
                )
            else:
                col_dir = data_path / col
                loader = make_loader(client_dir=col_dir, batch_size=1, patch_size=p_size, is_train=True)

            local_model = get_model(model_type=model_type).to(dev)
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()

            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

            col_samples = 0
            if round_batches is not None and round_batches > 0:
                # Fixed batch training
                batch_count = 0
                while batch_count < round_batches:
                    for x, y in loader:
                        if batch_count >= round_batches:
                            break
                        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        logits = local_model(x)
                        loss = loss_fn(logits, y.float())
                        loss.backward()
                        optimizer.step()
                        col_samples += x.size(0)
                        batch_count += 1
            else:
                num_epochs = max(1, int(round(round_epochs))) if round_epochs is not None else 1
                for _ in range(num_epochs):
                    for x, y in loader:
                        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        logits = local_model(x)
                        loss = loss_fn(logits, y.float())
                        loss.backward()
                        optimizer.step()
                        col_samples += x.size(0)

            client_states.append(local_model.state_dict())
            client_num_examples.append(col_samples)

        # 4. Aggregation
        if aggregation_method == "regsimagg":
            w_base, sim_w, sample_w, dists = base_weights(
                client_states, client_num_examples, distance_mode=distance_mode
            )
            if previous_global_state is not None and r > regularization_round:
                weights, deltas = regularize_weights(
                    w_base, client_states, previous_global_state
                )
            else:
                weights = w_base
                deltas = [0.0] * len(client_states)

            agg_state = aggregate_states(client_states, weights)
        else:
            # Simple FedAvg baseline
            weights = (torch.tensor(client_num_examples, dtype=torch.float32) / sum(client_num_examples)).numpy()
            agg_state = aggregate_states(client_states, weights)
            sim_w, sample_w, dists, deltas = weights, weights, [0.0] * len(weights), [0.0] * len(weights)

        # Update global model
        previous_global_state = global_model.state_dict()
        global_model.load_state_dict({k: torch.from_numpy(v) for k, v in agg_state.items()})

        # 5. Validation
        global_model.eval()
        val_dice_wt, val_dice_tc, val_dice_et = [], [], []

        with torch.no_grad():
            # Validate on a subset of cases
            val_cols = selected_cols
            for col in val_cols:
                if use_real_fets:
                    v_loader = make_loader(
                        data_root=train_path,
                        partition_csv=csv_path,
                        partition_id=col,
                        batch_size=1,
                        patch_size=p_size,
                        shuffle=False,
                        is_train=False,
                    )
                else:
                    col_dir = data_path / col
                    v_loader = make_loader(client_dir=col_dir, batch_size=1, patch_size=p_size, shuffle=False, is_train=False)

                for idx, (x, y) in enumerate(v_loader):
                    if idx >= 2:  # sample check for speed
                        break
                    x, y = x.to(dev), y.to(dev)
                    logits = global_model(x)
                    m = evaluate_subregions(logits, y)
                    val_dice_wt.append(m["dice_wt"])
                    val_dice_tc.append(m["dice_tc"])
                    val_dice_et.append(m["dice_et"])

        mean_wt = float(sum(val_dice_wt) / max(len(val_dice_wt), 1))
        mean_tc = float(sum(val_dice_tc) / max(len(val_dice_tc), 1))
        mean_et = float(sum(val_dice_et) / max(len(val_dice_et), 1))
        mean_avg = (mean_wt + mean_tc + mean_et) / 3.0

        elapsed = time.time() - round_start
        record = {
            "round": r,
            "selected_clients": selected_cols,
            "weights": [round(float(w), 4) for w in weights],
            "dice_wt": round(mean_wt, 4),
            "dice_tc": round(mean_tc, 4),
            "dice_et": round(mean_et, 4),
            "dice_avg": round(mean_avg, 4),
            "round_duration_sec": round(elapsed, 2),
        }
        round_records.append(record)
        logger.info("Round %d Evaluation: Dice WT=%.4f, TC=%.4f, ET=%.4f, Avg=%.4f (%.1fs)",
                    r, mean_wt, mean_tc, mean_et, mean_avg, elapsed)

    df_scores = pd.DataFrame(round_records)
    csv_path_out = out_dir / "experiment_results.csv"
    df_scores.to_csv(csv_path_out, index=False)
    logger.info("Saved results to %s in %.2fs total", csv_path_out, time.time() - start_time)

    # Save final model
    torch.save(global_model.state_dict(), str(out_dir / "best_model.pt"))

    # Run validation inference if requested
    if predict_val and val_data_dir is not None:
        val_path = Path(val_data_dir)
        if val_path.exists():
            val_out = out_dir / "val_predictions"
            effective_val_limit = val_limit if val_limit is not None else 5
            logger.info("Running post-experiment validation inference on %s (limit: %s)...", val_path, effective_val_limit)
            model_outputs_to_disc(
                data_path=val_path,
                output_path=val_out,
                native_model_path=str(out_dir / "best_model.pt"),
                device=device,
                model_type=model_type,
                limit=effective_val_limit,
            )
            logger.info("Validation predictions generated at %s", val_out)

    return df_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FeTS RegSimAgg Experiment")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, default=5, help="Total number of collaborators")
    parser.add_argument("--fraction", type=float, default=0.2, help="Fraction of clients per round (default 0.2 = 20%)")
    parser.add_argument("--method", type=str, default="regsimagg", choices=["regsimagg", "fedavg"])
    parser.add_argument("--distance", type=str, default="github_compat", choices=["github_compat", "paper_l1"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--train-data", type=str, default=r"C:\Users\syedh\Documents\BraT_Segmentation_FeTs2022\MICCAI_FeTS2022_TrainingData")
    parser.add_argument("--val-data", type=str, default=r"C:\Users\syedh\Documents\BraT_Segmentation_FeTs2022\MICCAI_FeTS2022_ValidationData")
    parser.add_argument("--partition-csv", type=str, default=r"C:\Users\syedh\Documents\BraT_Segmentation_FeTs2022\MICCAI_FeTS2022_TrainingData\partitioning_2.csv")
    parser.add_argument("--model-type", type=str, default="monai_unet", choices=["small_unet", "monai_unet"])
    parser.add_argument("--patch-size", type=int, default=64, help="Spatial patch dimension (64 for CPU, 96/128 for GPU)")
    parser.add_argument("--loss", type=str, default="dice", choices=["dice", "dice_ce"], help="Loss function (dice or dice_ce)")
    parser.add_argument("--batches", type=int, default=None, help="Fixed number of batches per round")
    parser.add_argument("--epochs", type=float, default=None, help="Epochs per client round")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-4, help="Learning rate (default 1e-4)")
    parser.add_argument("--predict-val", action="store_true", help="Generate NIfTI predictions on validation set after training")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit number of validation cases to infer (default 5)")
    args = parser.parse_args()

    df = run_challenge_experiment(
        aggregation_method=args.method,
        distance_mode=args.distance,
        rounds_to_train=args.rounds,
        num_clients=args.clients,
        fraction_train=args.fraction,
        learning_rate=args.lr,
        device=args.device,
        data_root=args.data_root,
        train_data_dir=args.train_data,
        val_data_dir=args.val_data,
        partition_csv=args.partition_csv,
        model_type=args.model_type,
        patch_size=args.patch_size,
        loss_type=args.loss,
        batches=args.batches,
        epochs=args.epochs,
        predict_val=args.predict_val,
        val_limit=args.val_limit,
    )
    print("\nExperiment Summary:\n", df[["round", "selected_clients", "weights", "dice_avg"]])
