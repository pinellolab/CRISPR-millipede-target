"""I/O and model-result extraction helpers for CRISPR-Millipede."""

from __future__ import annotations

import os
import pickle
from typing import Optional

from .models_inputs import MillipedeModelType


def load_latest_pickle(model_dir: str, label: str):
    """Load the most recent pickle matching ``label`` from ``model_dir``.

    Files are selected by prefix match and the latest file is chosen by mtime.
    """
    from .utils import display_all_pickle_versions

    candidates = display_all_pickle_versions(model_dir + "/", label)
    if not candidates:
        raise FileNotFoundError(f"No pickles found for label={label} in {model_dir}")

    candidates_with_time = [
        (name, os.path.getmtime(os.path.join(model_dir, name))) for name in candidates
    ]
    latest = max(candidates_with_time, key=lambda x: x[1])[0]
    path = os.path.join(model_dir, latest)

    with open(path, "rb") as handle:
        obj = pickle.load(handle)

    return obj, latest


def unwrap_selector(result_wrapper, model_type=MillipedeModelType.NORMAL_SIGMA_SCALED):
    """Extract a selector object from a model result wrapper."""
    return result_wrapper.millipede_model_specification_single_matrix_result[model_type]


def get_selector(
    model_group,
    spec_name: str,
    model_type=MillipedeModelType.NORMAL_SIGMA_SCALED,
):
    """Return selector for ``spec_name`` from a model group.

    Supported ``spec_name`` values:
    - ``joint``: joint experiment model
    - anything else: per-experiment model
    """
    if spec_name == "joint":
        result = model_group.millipede_model_specification_set_with_results[
            "joint_replicate_joint_experiment_models"
        ]
        wrapped = result.millipede_model_specification_result_input
        if isinstance(wrapped, list):
            wrapped = wrapped[0]
        return unwrap_selector(wrapped, model_type=model_type)

    result = model_group.millipede_model_specification_set_with_results[
        "joint_replicate_per_experiment_models"
    ]
    wrapped = result.millipede_model_specification_result_input
    if isinstance(wrapped, list):
        wrapped = wrapped[0]
    return unwrap_selector(wrapped, model_type=model_type)


def get_joint_selector(model_group, model_type=MillipedeModelType.NORMAL_SIGMA_SCALED):
    """Convenience wrapper for extracting joint-replicate per-experiment selector."""
    result = model_group.millipede_model_specification_set_with_results[
        "joint_replicate_per_experiment_models"
    ]
    wrapped = result.millipede_model_specification_result_input
    if isinstance(wrapped, list):
        wrapped = wrapped[0]
    return unwrap_selector(wrapped, model_type=model_type)
