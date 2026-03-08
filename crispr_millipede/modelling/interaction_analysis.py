"""Reusable interaction-analysis utilities for CRISPR-Millipede outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ThresholdResult:
    method: str
    threshold: float
    valid_pairs: List[Tuple[str, str]]
    pair_frequencies: Dict[Tuple[str, str], float]
    statistics: Dict[str, float]


def plot_interaction_pair_frequency_distribution(
    pair_frequencies: Dict[Tuple[str, str], float],
    threshold: float,
    title: str,
    output_path: str,
    bins: int = 50,
):
    """Plot interaction pair-frequency histogram with threshold marker."""
    freqs = np.array(list(pair_frequencies.values()), dtype=float)
    if len(freqs) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(freqs, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.4f}",
    )
    ax.set_xlabel("Read-weighted Co-editing Frequency (IoU)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_variant_position(variant_name: str) -> Optional[int]:
    """Extract genomic position from a variant string like ``160A>G``."""
    match = re.match(r"^(\d+)[ACGT]>[ACGT]$", str(variant_name))
    return int(match.group(1)) if match else None


def parse_variant(variant_name: str) -> Tuple[Optional[int], Optional[str]]:
    """Return (position, mutation) for variant strings like ``160A>G``."""
    match = re.match(r"^(\d+)([ACGT])>([ACGT])$", str(variant_name))
    if not match:
        return None, None
    return int(match.group(1)), f"{match.group(2)}>{match.group(3)}"


def classify_interaction_mechanism(row: pd.Series, pip_threshold: float = 0.1):
    """Classify interaction mechanism into Type A/B/C.

    Type C: Emergent (no significant main effects)
    Type B: De Novo (interaction sign differs from >=1 significant main effect)
    Type A: Reinforcing (interaction sign matches all significant main effects)
    """

    def _sign_label(x: float) -> str:
        if x > 0:
            return "+"
        if x < 0:
            return "-"
        return "0"

    var1_sig = row["var1_pip"] > pip_threshold
    var2_sig = row["var2_pip"] > pip_threshold

    int_sign = _sign_label(row["interaction_beta"])
    var1_sign = _sign_label(row["var1_beta"])
    var2_sign = _sign_label(row["var2_beta"])

    significant_terms = []
    if var1_sig:
        significant_terms.append(("var1", row["variant1"], var1_sign))
    if var2_sig:
        significant_terms.append(("var2", row["variant2"], var2_sign))

    if not significant_terms:
        return (
            "Type C: Emergent",
            "Neither component significant",
            "none",
            "none",
            "none",
        )

    differs = [name for _, name, s in significant_terms if s != int_sign]
    same = [name for _, name, s in significant_terms if s == int_sign]
    sig_names = [name for _, name, _ in significant_terms]

    if differs:
        detail = f"Opposite vs significant term(s): {', '.join(differs)}"
        return (
            "Type B: De Novo",
            detail,
            ", ".join(sig_names),
            ", ".join(differs),
            ", ".join(same) if same else "none",
        )

    detail = f"Same sign as all significant term(s): {', '.join(same)}"
    return (
        "Type A: Reinforcing",
        detail,
        ", ".join(sig_names),
        "none",
        ", ".join(same),
    )


def classify_interaction_table(df: pd.DataFrame, pip_threshold: float = 0.1) -> pd.DataFrame:
    """Add mechanism classification columns to an interaction table."""
    out = df.copy()
    classified = out.apply(lambda r: classify_interaction_mechanism(r, pip_threshold), axis=1)
    (
        out["mechanism_type"],
        out["mechanism_detail"],
        out["significant_independent_terms"],
        out["differs_from_significant_terms"],
        out["same_as_significant_terms"],
    ) = zip(*classified)
    return out


def build_joint_feature_tables(selector) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build PIP/Beta feature views from a selector.

    Returns:
    - no_intercept view
    - full feature view
    - intercept-only subset
    """
    pip = selector.pip.rename("PIP").to_frame()
    beta = selector.beta.rename("Beta").to_frame()

    no_intercept = (
        pip.join(beta, how="left")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("PIP", ascending=False)
        .reset_index(drop=True)
    )
    no_intercept["abs_beta"] = no_intercept["Beta"].abs()

    with_intercept = beta.join(pip, how="left").reset_index().rename(columns={"index": "feature"})
    with_intercept["PIP"] = with_intercept["PIP"].fillna(0.0)
    with_intercept["abs_beta"] = with_intercept["Beta"].abs()

    intercept_only = with_intercept[
        with_intercept["feature"].str.contains("intercept", case=False, na=False)
    ].copy()

    return no_intercept, with_intercept, intercept_only


def _compute_pair_frequencies(
    df: pd.DataFrame,
    variant_cols: List[str],
    reads_col: str,
) -> Dict[Tuple[str, str], float]:
    x = df[variant_cols].to_numpy(dtype=float)
    w = df[reads_col].to_numpy(dtype=float)

    weighted = x * w[:, None]
    intersection = x.T @ weighted
    variant_counts = weighted.sum(axis=0)
    union = variant_counts[:, None] + variant_counts[None, :] - intersection

    pair_freq = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0,
    )

    out = {}
    for i in range(len(variant_cols)):
        for j in range(i + 1, len(variant_cols)):
            out[(variant_cols[i], variant_cols[j])] = float(pair_freq[i, j])
    return out


def compute_interaction_pair_thresholds(
    loader,
    method: str = "fixed",
    threshold: float = 0.1,
    method_params: Optional[dict] = None,
) -> ThresholdResult:
    """Compute interaction-pair threshold from merged loader data.

    Methods:
    - fixed: use supplied threshold
    - percentile: keep top ``threshold`` fraction, where threshold in (0,1)
    - dynamic: median by default, or biotype-aware KDE intersection when available
    """
    if method_params is None:
        method_params = {}

    all_dfs = []
    for exp_dfs in loader.unprocessed_merged_experiment_df_list:
        for rep_df in exp_dfs:
            all_dfs.append(rep_df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    variant_cols = [c for c in merged_df.columns if ">" in c]
    reads_col = "#Reads_Presort" if "#Reads_Presort" in merged_df.columns else "#Reads_Presort_raw"

    pair_frequencies = _compute_pair_frequencies(merged_df, variant_cols, reads_col)

    freqs = np.array(list(pair_frequencies.values()))
    if len(freqs) == 0:
        return ThresholdResult(method=method, threshold=threshold, valid_pairs=[], pair_frequencies={}, statistics={})

    if method == "fixed":
        threshold_applied = float(threshold)
    elif method == "percentile":
        threshold_applied = float(np.percentile(freqs, (1.0 - threshold) * 100.0))
    elif method == "dynamic":
        threshold_applied = float(np.median(freqs))
        if method_params.get("use_biotype"):
            try:
                from scipy.optimize import fminbound
                from scipy.stats import gaussian_kde

                neighbor_freqs = []
                other_freqs = []
                for (v_i, v_j), freq in pair_frequencies.items():
                    pos_i, mut_i = parse_variant(v_i)
                    pos_j, mut_j = parse_variant(v_j)
                    is_neighbor = (
                        pos_i is not None
                        and pos_j is not None
                        and abs(pos_i - pos_j) <= 8
                        and mut_i == mut_j
                        and mut_i in ["A>G", "T>C"]
                    )
                    if is_neighbor:
                        neighbor_freqs.append(freq)
                    else:
                        other_freqs.append(freq)

                if len(neighbor_freqs) > 1 and len(other_freqs) > 1:
                    n_arr = np.array(neighbor_freqs)
                    o_arr = np.array(other_freqs)
                    kde_n = gaussian_kde(n_arr)
                    kde_o = gaussian_kde(o_arr)

                    x_min = max(float(n_arr.min()), float(o_arr.min()))
                    x_max = min(float(n_arr.max()), float(o_arr.max()))
                    if x_max > x_min:
                        x_test = np.linspace(x_min, x_max, 1000)
                        diff = kde_n(x_test) - kde_o(x_test)
                        sign_changes = np.where(np.diff(np.sign(diff)))[0]
                        if len(sign_changes) > 0:
                            idx = int(sign_changes[0])
                            x_left, x_right = x_test[idx], x_test[idx + 1]
                            res = fminbound(
                                lambda x: abs(float(kde_n(x) - kde_o(x))),
                                x_left,
                                x_right,
                                full_output=True,
                            )
                            threshold_applied = float(res[0])
            except Exception:
                # If scipy is unavailable or KDE fails, median remains the fallback.
                pass
    else:
        raise ValueError(f"Unknown method: {method}")

    valid_pairs = [
        (v_i, v_j)
        for (v_i, v_j), freq in pair_frequencies.items()
        if freq >= threshold_applied
    ]

    selected_freqs = [pair_frequencies[p] for p in valid_pairs] if valid_pairs else [0.0]
    stats = {
        "total_pairs": float(len(pair_frequencies)),
        "selected_pairs": float(len(valid_pairs)),
        "selection_rate": float(len(valid_pairs) / len(pair_frequencies)),
        "selected_freq_mean": float(np.mean(selected_freqs)),
        "selected_freq_median": float(np.median(selected_freqs)),
        "selected_freq_min": float(np.min(selected_freqs)),
        "selected_freq_max": float(np.max(selected_freqs)),
    }

    return ThresholdResult(
        method=method,
        threshold=threshold_applied,
        valid_pairs=valid_pairs,
        pair_frequencies=pair_frequencies,
        statistics=stats,
    )
