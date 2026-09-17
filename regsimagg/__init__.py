"""RegSimAgg: Regularized Similarity-based Aggregation for Federated Learning."""

from .collaborator_selector import (
    CollaboratorSelector,
    custom_percentage_collaborator_selector_without_repetition,
    custom_percentage_collaborator_selector,
    equal_partitions,
    all_collaborators_train,
    one_collaborator_on_odd_rounds,
)
from .hyperparameters import (
    constant_hyper_parameters,
    train_less_each_round,
    fixed_number_of_batches,
    get_hyperparameters_for_round,
)
from .regsimagg import (
    my_sum,
    paper_l1_distance,
    github_compat_distance,
    similarity_weights,
    sample_weights,
    base_weights,
    regularize_weights,
    reg_sim_weighted_average_layerwise,
    aggregate_states,
    fedavg_aggregate,
    clipped_aggregate,
    fedavgm_aggregate,
)
from .model import (
    Small3DUNet,
    build_fets_unet,
    get_model,
    convert_labels_to_subregions,
    convert_subregion_logits_to_labels,
    evaluate_subregions,
)
from .data import FeTSNiftiDataset, make_loader, generate_synthetic_fets_data
from .partitioner import parse_fets_partition_csv, make_partitions
from .strategy import RegSimAgg
from .inference import model_outputs_to_disc

__all__ = [
    "CollaboratorSelector",
    "custom_percentage_collaborator_selector_without_repetition",
    "custom_percentage_collaborator_selector",
    "equal_partitions",
    "all_collaborators_train",
    "one_collaborator_on_odd_rounds",
    "constant_hyper_parameters",
    "train_less_each_round",
    "fixed_number_of_batches",
    "get_hyperparameters_for_round",
    "my_sum",
    "paper_l1_distance",
    "github_compat_distance",
    "similarity_weights",
    "sample_weights",
    "base_weights",
    "regularize_weights",
    "reg_sim_weighted_average_layerwise",
    "aggregate_states",
    "fedavg_aggregate",
    "clipped_aggregate",
    "fedavgm_aggregate",
    "Small3DUNet",
    "build_fets_unet",
    "get_model",
    "convert_labels_to_subregions",
    "convert_subregion_logits_to_labels",
    "evaluate_subregions",
    "FeTSNiftiDataset",
    "make_loader",
    "generate_synthetic_fets_data",
    "parse_fets_partition_csv",
    "make_partitions",
    "RegSimAgg",
    "model_outputs_to_disc",
]
