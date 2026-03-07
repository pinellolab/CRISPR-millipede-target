"""
Mathematical utilities and data normalization functions for CRISPR-Millipede.

This module contains helper functions for decay modeling, exponential decay fitting,
count normalization across different experimental conditions, and pickle I/O utilities.
"""

import numpy as np
import pandas as pd
import pickle
from datetime import date
from os import listdir
from os.path import isfile, join
from typing import List, Optional


def decay_function(x, k, a, c=0.0, epsilon=0.01, decay_asymptote_offset=1):
    """
    Exponential decay function model.
    
    Exponential rate constant corresponding to epsilon at decay scale a.
    Shifted exponential decay toward asymptote.
    """
    b = -np.log(epsilon) / a
    # Shifted exponential decay toward asymptote
    c = c + decay_asymptote_offset
    return c + (k - c) * np.exp(-b * x)


def decay_function_2d(
    enriched_count,
    baseline_count,
    A1_parameter_2D,
    k1_parameter_enriched_2D,
    k1_parameter_baseline_2D,
    A2_parameter_2D,
    k2_parameter_enriched_2D,
    k2_parameter_baseline_2D,
    C_parameter_2D
):
    """
    Evaluate the fitted 2D *double exponential* decay surface.

    Parameters
    ----------
    enriched_count : float
        Enriched read depth (x)

    baseline_count : float
        Baseline read depth (y)

    A1_parameter_2D, A2_parameter_2D : float
        Amplitudes of the two exponential components.

    k1_parameter_enriched_2D, k2_parameter_enriched_2D : float
        Decay rates for enriched counts in components 1 and 2.

    k1_parameter_baseline_2D, k2_parameter_baseline_2D : float
        Decay rates for baseline counts in components 1 and 2.

    C_parameter_2D : float
        Asymptotic floor.

    Returns
    -------
    float
        Decay value at (x, y)
    """

    term1 = A1_parameter_2D * np.exp(
        -(k1_parameter_enriched_2D * enriched_count +
          k1_parameter_baseline_2D * baseline_count)
    )

    term2 = A2_parameter_2D * np.exp(
        -(k2_parameter_enriched_2D * enriched_count +
          k2_parameter_baseline_2D * baseline_count)
    )

    decay_minimum = 1

    return decay_minimum + term1 + term2 + C_parameter_2D


def normalize_counts(encoding_df: pd.DataFrame,
                     enriched_pop_df_reads_colname: str,
                     baseline_pop_df_reads_colname: str,
                     nucleotide_ids: List[str],
                     wt_normalization: bool,
                     total_normalization: bool,
                     presort_pop_df_reads_colname: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize read counts across different populations.
    
    TODO 5/15/23: Normalization is set to True always! Make it an input variable. 
                  Also, it should directly change the count rather than just the score
    TODO 5/15/23: Also, allow normalization either by library size or by WT reads. 
                  For now, will just do WT reads
    """
    # Original
    enriched_read_counts = encoding_df[enriched_pop_df_reads_colname]
    baseline_read_counts = encoding_df[baseline_pop_df_reads_colname]

    if presort_pop_df_reads_colname is not None:
        presort_read_counts = encoding_df[presort_pop_df_reads_colname]
    # IMPORTANT NOTE 5/15/23: Not updated the total_reads column since this 
    # column is used for the sigma_scale_factor

    # Perform normalization based on WT allele count
    if wt_normalization:
        wt_allele_df = encoding_df[encoding_df[nucleotide_ids].sum(axis=1) == 0]
        assert wt_allele_df.shape[0] == 1, f"No single WT allele present in encoding DF of shape {wt_allele_df.shape}"

        wt_enriched_read_count = wt_allele_df[enriched_pop_df_reads_colname][0]
        wt_baseline_read_count = wt_allele_df[baseline_pop_df_reads_colname][0]

        enriched_read_counts = enriched_read_counts * (wt_baseline_read_count / wt_enriched_read_count)
        baseline_read_counts = baseline_read_counts

        # Keep raw counts:
        encoding_df[enriched_pop_df_reads_colname + "_raw"] = encoding_df[enriched_pop_df_reads_colname]
        encoding_df[baseline_pop_df_reads_colname + "_raw"] = encoding_df[baseline_pop_df_reads_colname]

        encoding_df[enriched_pop_df_reads_colname] = enriched_read_counts
        encoding_df[baseline_pop_df_reads_colname] = baseline_read_counts

        if presort_pop_df_reads_colname is not None:
            wt_presort_read_count = wt_allele_df[presort_pop_df_reads_colname][0]
            presort_read_counts = presort_read_counts * (wt_baseline_read_count / wt_presort_read_count)
            encoding_df[presort_pop_df_reads_colname + "_raw"] = encoding_df[presort_pop_df_reads_colname]
            encoding_df[presort_pop_df_reads_colname] = presort_read_counts

    elif total_normalization:
        total_enriched_read_count = sum(enriched_read_counts)
        total_baseline_read_count = sum(baseline_read_counts)

        enriched_read_counts = enriched_read_counts * (total_baseline_read_count / total_enriched_read_count)
        baseline_read_counts = baseline_read_counts

        # Keep raw counts:
        encoding_df[enriched_pop_df_reads_colname + "_raw"] = encoding_df[enriched_pop_df_reads_colname]
        encoding_df[baseline_pop_df_reads_colname + "_raw"] = encoding_df[baseline_pop_df_reads_colname]

        encoding_df[enriched_pop_df_reads_colname] = enriched_read_counts
        encoding_df[baseline_pop_df_reads_colname] = baseline_read_counts

        if presort_pop_df_reads_colname is not None:
            total_presort_read_count = sum(presort_read_counts)
            presort_read_counts = presort_read_counts / total_presort_read_count
            encoding_df[presort_pop_df_reads_colname + "_raw"] = encoding_df[presort_pop_df_reads_colname]
            encoding_df[presort_pop_df_reads_colname] = presort_read_counts
    else:
        encoding_df[enriched_pop_df_reads_colname + "_raw"] = encoding_df[enriched_pop_df_reads_colname]
        encoding_df[baseline_pop_df_reads_colname + "_raw"] = encoding_df[baseline_pop_df_reads_colname]
        if presort_pop_df_reads_colname is not None:
            encoding_df[presort_pop_df_reads_colname + "_raw"] = encoding_df[presort_pop_df_reads_colname]

    # TODO 20240808: Implement size factor normalization - Zain has code
    return encoding_df


def save_or_load_pickle(directory, label, py_object=None, date_string=None):
    """
    Save or load a pickle file with date notation for caching.
    
    Parameters
    ----------
    directory : str
        Directory path for pickle file storage
    label : str
        Label/name for the pickle file
    py_object : object, optional
        Python object to save. If None, will load from existing pickle.
    date_string : str, optional
        Date string for versioning. If None, uses current date (YYYYMMDD format).
    
    Returns
    -------
    object
        If py_object is None, returns the loaded object from pickle file.
        Otherwise, saves the object and returns None.
    """
    if date_string is None:
        today = date.today()
        date_string = str(today.year) + ("0" + str(today.month) if today.month < 10 else str(today.month)) + str(today.day)
    
    filename = directory + label + "_" + date_string + '.pickle'
    print(filename)
    if py_object is None:
        with open(filename, 'rb') as handle:
            py_object = pickle.load(handle)
            return py_object
    else:
        with open(filename, 'wb') as handle:
            pickle.dump(py_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def display_all_pickle_versions(directory, label):
    """
    Retrieve all pickle files with a given label to identify available versions.
    
    Parameters
    ----------
    directory : str
        Directory path containing pickle files
    label : str
        Label/name prefix to search for
    
    Returns
    -------
    list
        List of filenames matching the label prefix
    """
    return [f for f in listdir(directory) if isfile(join(directory, f)) and label == f[:len(label)]]


def add_interaction_terms(df: pd.DataFrame,
                         nucleotide_id_cols: List[str],
                         coediting_frequency_threshold: float = 0.1,
                         read_count_colname: Optional[str] = None) -> pd.DataFrame:
    """
    Add pairwise interaction terms for variants that frequently co-edit (weighted by read counts).
    
    Interaction terms capture epistatic/combinatorial effects between variants by creating
    product features for variant pairs that co-edit above a specified frequency threshold.
    
    Co-editing frequency is calculated as read-weighted Intersection over Union (IoU):
        IoU = weighted_intersection / weighted_union
    where reads are weighted by total read counts per allele. This gives higher weight to
    abundant alleles and produces more robust co-editing estimates than unweighted counts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Design matrix dataframe containing variant features (columns with ">") and read counts
    nucleotide_id_cols : List[str]
        List of column names representing individual variant features
    coediting_frequency_threshold : float, default=0.1
        Minimum read-weighted co-editing frequency (0.0 to 1.0) required to create an
        interaction term. Pairs with weighted IoU >= threshold will get an interaction feature.
    read_count_colname : Optional[str], default=None
        Name of the normalized read-count column to use for weighting (for example,
        #Reads_Presort, #Reads_HbFHigh, or #Reads_HbFLow).
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with additional interaction term columns appended
        
    Notes
    -----
    - Interaction columns are named as "variant1_x_variant2" (lexicographically sorted)
    - Interaction values are computed as the product of the two variant indicators
    - Only variant pairs with read-weighted co-editing frequency >= threshold are included
    - Co-editing frequency uses read-weighted Intersection over Union (Jaccard similarity)
    """
    df = df.copy()
    
    # Extract variant columns from the dataframe
    variant_cols = [col for col in nucleotide_id_cols if col in df.columns]
    
    # Determine read count column from provided variable name (normalized column only)
    if read_count_colname is None:
        raise ValueError(
            "read_count_colname must be provided for read-weighted co-editing calculation."
        )

    if read_count_colname in df.columns:
        reads_col = read_count_colname
    else:
        raise ValueError(
            f"Read count column not found. Expected normalized column '{read_count_colname}' in dataframe columns."
        )
    
    # Binary variant matrix and read weights
    X = df[variant_cols].to_numpy(dtype=float)
    w = df[reads_col].to_numpy(dtype=float)
    
    # Weighted intersection counts: reads containing both variant_i and variant_j
    WX = X * w[:, None]  # Weight each variant by read count
    intersection_counts = X.T @ WX  # Pairwise weighted intersections
    
    # Weighted per-variant support: reads containing each variant
    variant_counts = WX.sum(axis=0)
    
    # Weighted union counts: reads containing variant_i OR variant_j
    union_counts = variant_counts[:, None] + variant_counts[None, :] - intersection_counts
    
    # Pairwise co-editing frequency = weighted intersection / weighted union (IoU/Jaccard)
    pair_freq = np.divide(
        intersection_counts,
        union_counts,
        out=np.zeros_like(intersection_counts, dtype=float),
        where=union_counts > 0,
    )
    
    # Build interaction terms from pairs above threshold
    interaction_terms_to_add = []
    
    for i in range(len(variant_cols)):
        for j in range(i + 1, len(variant_cols)):
            var1 = variant_cols[i]
            var2 = variant_cols[j]
            
            # Get read-weighted co-editing frequency from precomputed matrix
            coediting_freq = pair_freq[i, j]
            
            # Create interaction term if above threshold
            if coediting_freq >= coediting_frequency_threshold:
                # Sort variant names lexicographically for consistent naming
                sorted_vars = sorted([var1, var2])
                interaction_name = f"{sorted_vars[0]}_x_{sorted_vars[1]}"
                
                # Interaction value is product of the two variant values
                interaction_values = df[var1] * df[var2]
                
                interaction_terms_to_add.append((interaction_name, interaction_values))
    
    # Add all interaction terms to dataframe
    for interaction_name, interaction_values in interaction_terms_to_add:
        df[interaction_name] = interaction_values

    print(f"Added {len(interaction_terms_to_add)} interaction terms "
          f"(read-weighted IoU threshold: {coediting_frequency_threshold}, "
          f"using read column: {reads_col})")
    
    return df
