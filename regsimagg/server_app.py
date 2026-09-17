"""Flower ServerApp orchestrating RegSimAgg federated training."""

from __future__ import annotations
import json
import logging
from pathlib import Path
import torch
from flwr.app import ArrayRecord, ConfigRecord, ServerApp

from .model import get_model
from .strategy import RegSimAgg

logger = logging.getLogger(__name__)
app = ServerApp()


@app.main()
def main(grid, context):
    """Entry point for Flower ServerApp."""
    run_cfg = context.run_config

    num_rounds = int(run_cfg.get("num-server-rounds", 10))
    fraction_train = float(run_cfg.get("fraction-train", 0.2))
    fraction_evaluate = float(run_cfg.get("fraction-evaluate", 1.0))
    lr = float(run_cfg.get("learning-rate", 5e-5))
    reg_round = int(run_cfg.get("regularization-round", 10))
    github_compat = bool(run_cfg.get("use-github-compat-distance", True))
    layerwise = bool(run_cfg.get("layerwise-aggregation", False))
    agg_method = str(run_cfg.get("aggregation-method", "regsimagg"))
    selector_method = str(run_cfg.get("collaborator-selector", "without_repetition"))
    hp_schedule = str(run_cfg.get("hyperparameters-schedule", "constant"))
    model_type = str(run_cfg.get("model-type", "small_unet"))
    batches_per_round = run_cfg.get("batches-per-round", None)
    if batches_per_round is not None:
        batches_per_round = int(batches_per_round)
        if hp_schedule == "constant":
            hp_schedule = "fixed_batches"
    epochs_per_round = run_cfg.get("epochs-per-round", 1.0)
    if epochs_per_round is not None:
        epochs_per_round = float(epochs_per_round)

    model = get_model(model_type=model_type)
    arrays = ArrayRecord(model.state_dict())

    strategy = RegSimAgg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=max(1, int(grid.num_nodes * fraction_train)) if hasattr(grid, "num_nodes") else 1,
        min_evaluate_nodes=1,
        min_available_nodes=1,
        regularization_round=reg_round,
        github_compat_distance=github_compat,
        layerwise_aggregation=layerwise,
        aggregation_method=agg_method,
        collaborator_selector_method=selector_method,
        hyperparameters_schedule=hp_schedule,
        learning_rate=lr,
        epochs_per_round=epochs_per_round,
        batches_per_round=batches_per_round,
    )

    strategy.summary()

    initial_train_config = strategy.get_round_train_config(server_round=0)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=initial_train_config,
        num_rounds=num_rounds,
    )

    # Save final model
    output_model_path = Path("regsimagg_final.pt")
    torch.save(result.arrays.to_torch_state_dict(), str(output_model_path))
    logger.info("Saved final global model to %s", output_model_path)

    # Save history of weights and metrics for reproduction verification
    if strategy.history_records:
        history_path = Path("regsimagg_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(strategy.history_records, f, indent=2)
        logger.info("Saved aggregation history records to %s", history_path)
