import numpy as np
from scipy.optimize import curve_fit
import torch
from millipede import NormalLikelihoodVariableSelector
from millipede import BinomialLikelihoodVariableSelector
from millipede import NegativeBinomialLikelihoodVariableSelector
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools
import crispr_shrinkage

import logging

from os.path import exists

from dataclasses import dataclass
from typing import Union, List, Mapping, Tuple, Optional
from functools import partial
from typeguard import typechecked
from enum import Enum
from collections import defaultdict

import re
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages

from .models_inputs import *

from .pydeseq import run_pydeseq2

def decay_function(x, k, a, c=1.0, epsilon=0.01):
    # Exponential rate constant corresponding to epsilon at decay scale a
    b = -np.log(epsilon) / a
    # Shifted exponential decay toward asymptote
    decay_minimum = 1
    return decay_minimum + c + (k - c) * np.exp(-b * x)

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
                          presort_pop_df_reads_colname: Optional[str]=None) -> pd.DataFrame:
        # TODO 5/15/23: Normalization is set to True always! Make it an input variable. Also, it should directly change the count rather than just the score
        # TODO 5/15/23: Also, allow normalization either by library size or by WT reads. For now, will just do WT reads
        
        # Original
        enriched_read_counts = encoding_df[enriched_pop_df_reads_colname]
        baseline_read_counts = encoding_df[baseline_pop_df_reads_colname]
        
        if presort_pop_df_reads_colname is not None:
            presort_read_counts = encoding_df[presort_pop_df_reads_colname]
        # IMPORTANT NOTE 5/15/23: Not updated the total_reads column since this column is used for the sigma_scale_factor
        
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
            
            enriched_read_counts = enriched_read_counts / total_enriched_read_count
            baseline_read_counts = baseline_read_counts / total_baseline_read_count
            
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

@dataclass 
class MillipedeInputDataLoader:
    data_directory: str
    enriched_pop_fn_experiment_list: List[str]
    enriched_pop_df_reads_colname: str
    baseline_pop_fn_experiment_list: List[str]
    baseline_pop_df_reads_colname: str
    experiment_labels: List[str]
    reps: Optional[List[List[int]]]
    presort_pop_fn_experiment_list: Optional[List[str]] = None
    presort_pop_df_reads_colname: Optional[str] = None
    unprocessed_merged_experiment_df_list: Optional[List[List[pd.DataFrame]]] = None

    """
        Generates the MillipedeInputData
    """
    def __post_init__(self):
        '''
            Input validation
        '''
        print("Performing initial input validation checks...")
        assert len(self.reps) > 0, "reps list must have length > 0"
        for fn in self.enriched_pop_fn_experiment_list + self.baseline_pop_fn_experiment_list:
            assert "{}" in fn, '"{}" must be in present in filename to designate replicate ID for string formating, original filename: ' + fn
        assert len(self.enriched_pop_fn_experiment_list) > 0, "Length of list enriched_pop_fn_experiment_list must not be 0"
        assert len(self.baseline_pop_fn_experiment_list) > 0, "Length of list baseline_pop_fn_experiment_list must not be 0"
        assert len(self.experiment_labels) > 0, "Length of list baseline_pop_fn_experiment_list must not be 0"
        assert len(self.enriched_pop_fn_experiment_list) == len(self.baseline_pop_fn_experiment_list)  == len(self.experiment_labels), "Length of enriched_pop_fn_experiment_list, baseline_pop_fn_experiment_list, and experiment_labels must be same length"

        if self.presort_pop_fn_experiment_list is not None:
            for fn in self.presort_pop_fn_experiment_list:
                assert "{}" in fn, '"{}" must be in present in filename to designate replicate ID for string formating, original filename: ' + fn
            assert len(self.presort_pop_fn_experiment_list) > 0, "Length of list presort_pop_fn_experiment_list must not be 0"
            assert len(self.experiment_labels) == len(self.presort_pop_fn_experiment_list), "Length of enriched_pop_fn_experiment_list, baseline_pop_fn_experiment_list, and experiment_labels must be same length"

        print("Passed validation.")
    
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            #
            # Process the replicate dataframes:
            # type List[pd.DataFrame] if relicates are combined
            # type List[List[pd.DataFrame]] if replicates are separate
            #
            unprocessed_merged_experiment_df_list: List[List[pd.DataFrame]] = []
            # Iterate through the experiments
            for experiment_index in range(len(self.enriched_pop_fn_experiment_list)):
                # Get the enriched_population and baseline_population for the experiment
                enriched_pop_exp_fn = self.enriched_pop_fn_experiment_list[experiment_index]
                baseline_pop_exp_fn = self.baseline_pop_fn_experiment_list[experiment_index]
                
                presort_pop_exp_fn = None
                if self.presort_pop_fn_experiment_list is not None:
                    presort_pop_exp_fn = self.presort_pop_fn_experiment_list[experiment_index]
            
                # Iterate through each replicate of the experiment
                # type List[pd.DataFrame] if relicates are combined
                # type List[List[pd.DataFrame]] if replicates are separate
                unprocessed_exp_merged_rep_df_list: List[pd.DataFrame] = []
                for rep in self.reps[experiment_index]:
                    '''
                        Check file directories
                    '''
                    enriched_pop_full_fn_exp_rep = (self.data_directory + '/' + enriched_pop_exp_fn).format(rep)
                    baseline_pop_full_fn_exp_rep = (self.data_directory + '/' + baseline_pop_exp_fn).format(rep)
                    
                    presort_pop_full_fn_exp_rep = None
                    if presort_pop_exp_fn is not None:
                        presort_pop_full_fn_exp_rep = (self.data_directory + '/' + presort_pop_exp_fn).format(rep)
                        assert exists(presort_pop_full_fn_exp_rep), "File not found: {}".format(presort_pop_full_fn_exp_rep)

                    assert exists(enriched_pop_full_fn_exp_rep), "File not found: {}".format(enriched_pop_full_fn_exp_rep)
                    assert exists(baseline_pop_full_fn_exp_rep), "File not found: {}".format(baseline_pop_full_fn_exp_rep)

                    '''
                        Read in dataframes
                    '''
                    enriched_pop_exp_rep_df = pd.read_csv(enriched_pop_full_fn_exp_rep, sep='\t').fillna(value=0.0)
                    enriched_pop_nt_columns = [col for col in enriched_pop_exp_rep_df.columns if ">" in col]
                    enriched_pop_exp_rep_df = enriched_pop_exp_rep_df[enriched_pop_nt_columns + [self.enriched_pop_df_reads_colname]]


                    baseline_pop_exp_rep_df = pd.read_csv(baseline_pop_full_fn_exp_rep, sep='\t').fillna(value=0.0)
                    baseline_pop_nt_columns = [col for col in baseline_pop_exp_rep_df.columns if ">" in col]
                    baseline_pop_exp_rep_df = baseline_pop_exp_rep_df[baseline_pop_nt_columns + [self.baseline_pop_df_reads_colname]]

                    assert set(enriched_pop_nt_columns) == set(baseline_pop_nt_columns), "Nucleotide columns between enriched and baseline dataframes must be equivalent - are these screening the same regions?"
                    
                    presort_pop_exp_rep_df = None
                    if presort_pop_full_fn_exp_rep is not None:
                        presort_pop_exp_rep_df = pd.read_csv(presort_pop_full_fn_exp_rep, sep='\t').fillna(value=0.0)
                        presort_pop_nt_columns = [col for col in presort_pop_exp_rep_df.columns if ">" in col]
                        presort_pop_exp_rep_df = presort_pop_exp_rep_df[presort_pop_nt_columns + [self.presort_pop_df_reads_colname]]
                        assert set(enriched_pop_nt_columns) == set(presort_pop_nt_columns), "Nucleotide columns between presort and the  enriched/baseline dataframes must be equivalent - are these screening the same regions?"


                    
                    nucleotide_ids = enriched_pop_nt_columns

                    # Concat the enriched and baseline population dataframes together
                    if presort_pop_full_fn_exp_rep is not None:
                        unprocessed_merged_exp_rep_df: pd.DataFrame = pd.concat([enriched_pop_exp_rep_df, baseline_pop_exp_rep_df, presort_pop_exp_rep_df]).groupby(nucleotide_ids, as_index=False).sum()
                    else:
                        unprocessed_merged_exp_rep_df: pd.DataFrame = pd.concat([enriched_pop_exp_rep_df, baseline_pop_exp_rep_df]).groupby(nucleotide_ids, as_index=False).sum()

                    unprocessed_exp_merged_rep_df_list.append(unprocessed_merged_exp_rep_df)
                unprocessed_merged_experiment_df_list.append(unprocessed_exp_merged_rep_df_list)
                self.unprocessed_merged_experiment_df_list = unprocessed_merged_experiment_df_list



    def plot_replicate_correlation(self) -> List[pd.DataFrame]:
        merged_df_experiment_list: List[pd.DataFrame] = []
        unprocessed_merged_experiment_df_list = self.unprocessed_merged_experiment_df_list
        for experiment_i, unprocessed_exp_merged_rep_df_list in enumerate(unprocessed_merged_experiment_df_list):
            print(f"Experiment {experiment_i}")
            merged_df = unprocessed_exp_merged_rep_df_list[0]
            merge_column_names = [col for col in merged_df.columns if ">" in col]
            
            # Add rep0 suffix
            def add_suffix(df, rep):
                return df.rename(
                    columns={col: f"{col}_rep{rep}" for col in df.columns if col not in merge_column_names}
                )
                
            merged_df = add_suffix(merged_df, 0)
            
            # Loop through the list of DataFrames and merge iteratively
            for replicate_i, unprocessed_merged_exp_rep_df in enumerate(unprocessed_exp_merged_rep_df_list[1:], start=1):
                unprocessed_merged_exp_rep_df = add_suffix(unprocessed_merged_exp_rep_df, replicate_i)
                print(f"Merging replicate {replicate_i}")
                merged_df = merged_df.merge(
                    unprocessed_merged_exp_rep_df,
                    on=merge_column_names,
                    how="outer"
                )
            merged_df = merged_df.fillna(0)


            num_pairwise_reps = len(self.reps)**2
            fig, axes = plt.subplots(num_pairwise_reps,2, figsize=(12, num_pairwise_reps*6))
            axes_row_i = 0
            for rep_i in self.reps[experiment_i]:
                for rep_j in self.reps[experiment_i]:
                    enriched_rep_i_colname = self.enriched_pop_df_reads_colname + f"_rep{rep_i}"
                    enriched_rep_j_colname = self.enriched_pop_df_reads_colname + f"_rep{rep_j}"
                    baseline_rep_i_colname = self.baseline_pop_df_reads_colname + f"_rep{rep_i}"
                    baseline_rep_j_colname = self.baseline_pop_df_reads_colname + f"_rep{rep_j}"
                    
                    axes[axes_row_i, 0].scatter(np.log2(merged_df[enriched_rep_i_colname]), np.log2(merged_df[enriched_rep_j_colname]), alpha=0.1)
                    axes[axes_row_i, 0].set_xlabel(f"Log2({enriched_rep_i_colname})")
                    axes[axes_row_i, 0].set_ylabel(f"Log2({enriched_rep_j_colname})")
                    axes[axes_row_i, 0].set_title(f"Enriched: Rep {rep_i} vs. Rep {rep_j}")
                    
                    axes[axes_row_i, 1].scatter(np.log2(merged_df[baseline_rep_i_colname]), np.log2(merged_df[baseline_rep_j_colname]), alpha=0.1)
                    axes[axes_row_i, 1].set_xlabel(f"Log2({baseline_rep_i_colname})")
                    axes[axes_row_i, 1].set_ylabel(f"Log2({baseline_rep_j_colname})")
                    axes[axes_row_i, 1].set_title(f"Baseline: Rep {rep_i} vs. Rep {rep_j}")
                    axes_row_i = axes_row_i + 1
            plt.show()

            merged_df_experiment_list.append(merged_df)
        return merged_df_experiment_list
        


    def plot_binned_reads_by_score_standard_deviation(
            self, bounded_score=True, ymax=500, bin_width=10, figsize_width=12, figsize_height=10, score_psuedocount = 1e-9
        ):

        # -----------------------------------------------------------
        # Helper: compute mean(|score|), n, SE per bin
        # -----------------------------------------------------------
        def compute_mean_abs_error(df, bin_col):
            grouped = df.groupby(bin_col)['score']
            mean_abs = grouped.apply(lambda x: np.mean(np.abs(x)))
            n = grouped.size()
            se = mean_abs / np.sqrt(n)
            return pd.DataFrame({
                bin_col: mean_abs.index,
                "mean_abs": mean_abs.values,
                "n": n.values,
                "se": se.values
            })

        # -----------------------------------------------------------
        # Helper: xtick labels "(n=123) 0–10"
        # -----------------------------------------------------------
        def make_xticklabels_with_n(interval_index, n_values):
            labels = []
            for interval, n in zip(interval_index.astype(str), n_values):
                clean = interval.replace("(", "").replace("]", "")
                L, R = clean.split(",")
                labels.append(f"(n={n}) {L.strip()}–{R.strip()}")
            return labels

        # -----------------------------------------------------------
        # Fit using binned midpoints → returns x_fit, y_fit, params
        # -----------------------------------------------------------
        def exp_decay_fit_binned(x_bins, y_bins, n_bins, cap_percentile=90):
            mask = (~np.isnan(x_bins)) & (~np.isnan(y_bins))
            x = x_bins[mask]
            y = y_bins[mask]
            w = n_bins[mask]

            # cap dominating bins
            cap = np.percentile(w, cap_percentile)
            w = np.minimum(w, cap)
            sigma = 1 / np.sqrt(w)

            # model is A exp(-B x) + C
            def model(x, A, B, C):
                return A * np.exp(-B * x) + C

            try:
                popt, _ = curve_fit(
                    model, x, y,
                    sigma=sigma,
                    absolute_sigma=True,
                    bounds=(0, np.inf),
                    maxfev=20000
                )
                return x, model(x, *popt), popt
            except RuntimeError:
                return x, y, (np.nan, np.nan, np.nan)

        # -----------------------------------------------------------
        # Convert (A, B, C) → (k, a, c)
        # -----------------------------------------------------------
        def convert_params(popt, epsilon=0.01):
            A, B, C = popt
            if np.any(np.isnan(popt)):
                return np.nan, np.nan, np.nan
            a = -np.log(epsilon) / B if B > 0 else np.nan
            k = A + C
            c = C
            return k, a, c

        # -----------------------------------------------------------
        # STORAGE: nested experiment × replicate lists
        # -----------------------------------------------------------
        k_parameter_enriched = []
        a_parameter_enriched = []
        c_parameter_enriched = []

        k_parameter_baseline = []
        a_parameter_baseline = []
        c_parameter_baseline = []

        k_parameter_enriched2D = []
        k_parameter_baseline2D = []
        A1_parameter2D = []
        kX1_parameter2D = []
        kY1_parameter2D = []
        A2_parameter2D = []
        kX2_parameter2D = []
        kY2_parameter2D = []
        C_parameter2D = []

        # -----------------------------------------------------------
        # MAIN LOOP
        # -----------------------------------------------------------
        unprocessed_merged_experiment_df_list_copy = []

        for experiment_i, unprocessed_exp_merged_rep_df_list in enumerate(self.unprocessed_merged_experiment_df_list):

            # Start lists for this experiment
            k_parameter_enriched.append([])
            a_parameter_enriched.append([])
            c_parameter_enriched.append([])

            k_parameter_baseline.append([])
            a_parameter_baseline.append([])
            c_parameter_baseline.append([])

            k_parameter_enriched2D.append([])
            k_parameter_baseline2D.append([])
            A1_parameter2D.append([])
            kX1_parameter2D.append([])
            kY1_parameter2D.append([])
            A2_parameter2D.append([])
            kX2_parameter2D.append([])
            kY2_parameter2D.append([])
            C_parameter2D.append([])

            unprocessed_exp_merged_rep_df_list_copy = []

            for replicate_i, df_raw in enumerate(unprocessed_exp_merged_rep_df_list):

                df = df_raw.copy()
                nt_columns = [col for col in df.columns if ">" in col]

                # normalize reads
                df_norm = normalize_counts(
                    df,
                    self.enriched_pop_df_reads_colname,
                    self.baseline_pop_df_reads_colname,
                    nt_columns,
                    True,
                    False
                )

                # LFC score = log2(high / low)
                high_vals = df[self.enriched_pop_df_reads_colname]
                low_vals  = df[self.baseline_pop_df_reads_colname]
                

                if bounded_score:
                    df_norm["score"] = ( high_vals - low_vals ) / ( high_vals + low_vals + score_psuedocount)
                else:
                    df_norm["score"] = np.log2((high_vals + score_psuedocount) / (low_vals + score_psuedocount))

                # binning
                bins = np.arange(0, ymax, bin_width)
                df_norm['enriched_bins'] = pd.cut(df[self.enriched_pop_df_reads_colname+"_raw"], bins)
                df_norm['baseline_bins'] = pd.cut(df[self.baseline_pop_df_reads_colname+"_raw"], bins)

                enriched_stats = compute_mean_abs_error(df_norm, 'enriched_bins')
                baseline_stats = compute_mean_abs_error(df_norm, 'baseline_bins')

                enriched_mid = np.array([(iv.left + iv.right)/2 for iv in enriched_stats['enriched_bins']])
                baseline_mid = np.array([(iv.left + iv.right)/2 for iv in baseline_stats['baseline_bins']])

                # 1D FITTING
                x_enr_fit, y_enr_fit, enr_params = exp_decay_fit_binned(
                    enriched_mid,
                    enriched_stats['mean_abs'].values,
                    enriched_stats['n'].values
                )
                x_bas_fit, y_bas_fit, bas_params = exp_decay_fit_binned(
                    baseline_mid,
                    baseline_stats['mean_abs'].values,
                    baseline_stats['n'].values
                )

                # SORT FOR PLOTTING
                enr_idx = np.argsort(x_enr_fit)
                x_enr_fit, y_enr_fit = x_enr_fit[enr_idx], y_enr_fit[enr_idx]

                bas_idx = np.argsort(x_bas_fit)
                x_bas_fit, y_bas_fit = x_bas_fit[bas_idx], y_bas_fit[bas_idx]

                # STORE 1D FIT PARAMETERS
                k_e, a_e, c_e = convert_params(enr_params)
                k_b, a_b, c_b = convert_params(bas_params)

                k_parameter_enriched[experiment_i].append(k_e)
                a_parameter_enriched[experiment_i].append(a_e)
                c_parameter_enriched[experiment_i].append(c_e)

                k_parameter_baseline[experiment_i].append(k_b)
                a_parameter_baseline[experiment_i].append(a_b)
                c_parameter_baseline[experiment_i].append(c_b)

                # ---------------------------------------------------
                # PLOTTING 1D
                # ---------------------------------------------------
                fig, axes = plt.subplots(2, 1, figsize=(figsize_width, figsize_height))
                plt.subplots_adjust(hspace=0.55)

                axes[0].errorbar(
                    enriched_mid,
                    enriched_stats['mean_abs'],
                    yerr=enriched_stats['se'],
                    fmt='o', markersize=4, capsize=3
                )
                axes[0].plot(x_enr_fit, y_enr_fit, color='red', linewidth=3)
                axes[0].set_ylabel("Mean |score|")
                axes[0].set_title(f"Smooth monotone decreasing fit (binned) vs {self.enriched_pop_df_reads_colname}\n"
                                f"Experiment {experiment_i}, Replicate {replicate_i}", fontsize=12)
                axes[0].set_xticks(enriched_mid)
                axes[0].set_xticklabels(make_xticklabels_with_n(enriched_stats['enriched_bins'], enriched_stats['n']),
                                        rotation=45, ha='right', fontsize=6)
                axes[0].set_xlim(0, ymax)

                axes[1].errorbar(
                    baseline_mid,
                    baseline_stats['mean_abs'],
                    yerr=baseline_stats['se'],
                    fmt='o', markersize=4, capsize=3
                )
                axes[1].plot(x_bas_fit, y_bas_fit, color='red', linewidth=3)
                axes[1].set_ylabel("Mean |score|")
                axes[1].set_title(f"Smooth monotone decreasing fit (binned) vs {self.baseline_pop_df_reads_colname}\n"
                                f"Experiment {experiment_i}, Replicate {replicate_i}", fontsize=12)
                axes[1].set_xticks(baseline_mid)
                axes[1].set_xticklabels(make_xticklabels_with_n(baseline_stats['baseline_bins'], baseline_stats['n']),
                                        rotation=45, ha='right', fontsize=6)
                axes[1].set_xlim(0, ymax)
                plt.show()

                # ---------------------------------------------------
                # 2D HEATMAP AND DOUBLE EXP FIT (monotone)
                # ---------------------------------------------------
                heatmap_df = df_norm[['score', 'enriched_bins', 'baseline_bins']].dropna()
                pivot = heatmap_df.pivot_table(
                    index='baseline_bins',
                    columns='enriched_bins',
                    values='score',
                    aggfunc=lambda x: np.std(x)
                )

                enr_intervals = pivot.columns
                bas_intervals = pivot.index

                # midpoints & edges
                enr_mid_hm = np.array([(iv.left + iv.right)/2 for iv in enr_intervals])
                bas_mid_hm = np.array([(iv.left + iv.right)/2 for iv in bas_intervals])
                enr_edges = np.array([iv.left for iv in enr_intervals] + [enr_intervals[-1].right])
                bas_edges = np.array([iv.left for iv in bas_intervals] + [bas_intervals[-1].right])

                heatmap_vals_full = np.full((len(bas_mid_hm), len(enr_mid_hm)), np.nan)
                weights2d_full = np.zeros_like(heatmap_vals_full)
                for i, bi in enumerate(bas_intervals):
                    for j, ej in enumerate(enr_intervals):
                        cell = heatmap_df[(heatmap_df['baseline_bins']==bi) & (heatmap_df['enriched_bins']==ej)]
                        if len(cell) > 0:
                            heatmap_vals_full[i,j] = np.std(cell['score'])
                            weights2d_full[i,j] = len(cell)

                Xf, Yf = np.meshgrid(enr_mid_hm, bas_mid_hm)
                Zf = heatmap_vals_full
                mask_fit = ~np.isnan(Zf)
                X_fit2d, Y_fit2d, Z_fit2d, sigma2d = Xf[mask_fit], Yf[mask_fit], Zf[mask_fit], 1/np.sqrt(weights2d_full[mask_fit]+1e-8)

                # double exponential, monotone
                def double_exp2d(coords, A1, kX1, kY1, A2, kX2, kY2, C):
                    x, y = coords
                    return (A1 * np.exp(-kX1 * x - kY1 * y) +
                            A2 * np.exp(-kX2 * x - kY2 * y) +
                            C)

                p0 = [np.nanmax(Z_fit2d)/2, 0.01, 0.01, np.nanmax(Z_fit2d)/2, 0.01, 0.01, np.nanmin(Z_fit2d)]
                lower_bounds = [0, 1e-6, 1e-6, 0, 1e-6, 1e-6, 0]
                upper_bounds = [np.inf]*7

                try:
                    popt2d, _ = curve_fit(
                        double_exp2d,
                        (X_fit2d, Y_fit2d),
                        Z_fit2d,
                        sigma=sigma2d,
                        p0=p0,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=30000
                    )
                except RuntimeError:
                    popt2d = [np.nan]*7

                Z_fit_surface = double_exp2d((Xf, Yf), *popt2d)

                # STORE 2D PARAMETERS
                A1_parameter2D[experiment_i].append(popt2d[0])
                kX1_parameter2D[experiment_i].append(popt2d[1])
                kY1_parameter2D[experiment_i].append(popt2d[2])
                A2_parameter2D[experiment_i].append(popt2d[3])
                kX2_parameter2D[experiment_i].append(popt2d[4])
                kY2_parameter2D[experiment_i].append(popt2d[5])
                C_parameter2D[experiment_i].append(popt2d[6])

                k_parameter_enriched2D[experiment_i].append(popt2d[0]+popt2d[3]+popt2d[6])
                k_parameter_baseline2D[experiment_i].append(popt2d[0]+popt2d[3]+popt2d[6])

                # 2D plotting
                fig, ax = plt.subplots(1, 2, figsize=(14, 6))
                cmap = plt.cm.viridis
                cmap.set_bad(color='white')

                im0 = ax[0].pcolormesh(enr_edges, bas_edges, Zf, shading='auto', cmap=cmap)
                ax[0].set_title('Empirical 2D Histogram')
                ax[0].set_xlabel(f"{self.enriched_pop_df_reads_colname} (read bin)")
                ax[0].set_ylabel(f"{self.baseline_pop_df_reads_colname} (read bin)")
                plt.colorbar(im0, ax=ax[0])

                im1 = ax[1].pcolormesh(enr_edges, bas_edges, Z_fit_surface, shading='auto', cmap=cmap)
                ax[1].set_title('Fitted 2D Double Exponential (monotone)')
                ax[1].set_xlabel(f"{self.enriched_pop_df_reads_colname} (read bin)")
                ax[1].set_ylabel(f"{self.baseline_pop_df_reads_colname} (read bin)")
                plt.colorbar(im1, ax=ax[1])

                plt.show()
                unprocessed_exp_merged_rep_df_list_copy.append(df)

            unprocessed_merged_experiment_df_list_copy.append(unprocessed_exp_merged_rep_df_list_copy)

        # -----------------------------------------------------------
        # RETURN STRUCTURED DICTIONARY
        # -----------------------------------------------------------
        return {
            "k_parameter_enriched": k_parameter_enriched,
            "a_parameter_enriched": a_parameter_enriched,
            "c_parameter_enriched": c_parameter_enriched,
            "k_parameter_baseline": k_parameter_baseline,
            "a_parameter_baseline": a_parameter_baseline,
            "c_parameter_baseline": c_parameter_baseline,
            "k_parameter_enriched2D": k_parameter_enriched2D,
            "k_parameter_baseline2D": k_parameter_baseline2D,
            "A1_parameter_2D": A1_parameter2D,
            "k1_parameter_enriched_2D": kX1_parameter2D,
            "k1_parameter_baseline_2D": kY1_parameter2D,
            "A2_parameter_2D": A2_parameter2D,
            "k2_parameter_enriched_2D": kX2_parameter2D,
            "k2_parameter_baseline_2D": kY2_parameter2D,
            "C_parameter_2D": C_parameter2D
        }



        
# TODO 20221019: Include presort in the filtering, so therefore must also take presort fn as input
@dataclass
class MillipedeInputDataExperimentalGroup:
    millipede_model_specification_set: Mapping[str, MillipedeModelSpecification]
    millipede_input_data_loader: MillipedeInputDataLoader
    remove_empty_features: bool = True

    """
        Generates the MillipedeInputData objects provided MillipedeModelSpecifications and other relevant parameters such as filepaths to the data tables, read thresholds, and labels.
    """
    def __post_init__(self):
                 
        '''
            Input validation
        '''
        assert self.millipede_input_data_loader.unprocessed_merged_experiment_df_list is not None, "millipede_input_data_loader.unprocessed_merged_experiment_df_list is None, make sure that you load data using MillipedeInputDataLoader class"
        print("Passed validation.")
        
        
        __get_data_partial = partial(
            self.__get_data,
            millipede_input_data_loader=self.millipede_input_data_loader, 
        )
        # This will be the variable containing the final dictionary with input design matrix for all specifications
        millipede_model_specification_set_with_data: Mapping[str, Tuple[MillipedeModelSpecification, MillipedeInputData]] = dict()
        
        # Helpful note: This retrieves the unique set of input design matrix to generate based on the provided model specifications (specifically the unique set of replicate merge strategy, experiment merge strategy, and other specifications as the model input data only varies based on these criteria)
        millipede_design_matrix_set: Mapping[Tuple[MillipedeReplicateMergeStrategy, MillipedeExperimentMergeStrategy, MillipedeCutoffSpecification, MillipedeShrinkageInput, MillipedeDesignMatrixProcessingSpecification], List[str]] = self.__determine_full_design_matrix_set(self.millipede_model_specification_set)
        self.__millipede_design_matrix_set = millipede_design_matrix_set
        
        # Below generates the input data and assigns to corresponding model specifications
        merge_strategy_and_cutoff_tuple: Tuple[MillipedeReplicateMergeStrategy, MillipedeExperimentMergeStrategy, MillipedeCutoffSpecification, MillipedeShrinkageInput, MillipedeDesignMatrixProcessingSpecification]
        millipede_model_specification_id_list: List[str]
        for merge_strategy_and_cutoff_tuple, millipede_model_specification_id_list in millipede_design_matrix_set.items():
            # Generate input data - most computationally intensive task in this method
            print("Retrieving data for\n\tReplicate Merge Strategy: {} \n\tExperiment Merge Strategy {}\n\tCutoff: {}\n\tMatrixProcessing: {}". format(merge_strategy_and_cutoff_tuple[0], merge_strategy_and_cutoff_tuple[1], merge_strategy_and_cutoff_tuple[2], merge_strategy_and_cutoff_tuple[4]))
            millipede_input_data: MillipedeInputData = __get_data_partial(
                replicate_merge_strategy=merge_strategy_and_cutoff_tuple[0], 
                experiment_merge_strategy=merge_strategy_and_cutoff_tuple[1],
                cutoff_specification=merge_strategy_and_cutoff_tuple[2],
                shrinkage_input=merge_strategy_and_cutoff_tuple[3],
                design_matrix_processing_specification=merge_strategy_and_cutoff_tuple[4]
            )
                
            # Assign input data to corresponding model specifications
            for millipede_model_specification_id in millipede_model_specification_id_list:
                millipede_model_specification_set_with_data[millipede_model_specification_id] = (self.millipede_model_specification_set[millipede_model_specification_id], millipede_input_data)
                
        self.millipede_model_specification_set_with_data = millipede_model_specification_set_with_data

    """ 
    Function to process the encoding dataframe (from encode pipeline script) and create design matrix for milliped
    """
    def __get_data(self, 
                   millipede_input_data_loader: MillipedeInputDataLoader, 
                   replicate_merge_strategy:MillipedeReplicateMergeStrategy, 
                   experiment_merge_strategy:MillipedeExperimentMergeStrategy,
                   cutoff_specification: MillipedeCutoffSpecification,
                   design_matrix_processing_specification: MillipedeDesignMatrixProcessingSpecification,
                   shrinkage_input: Union[MillipedeShrinkageInput, None]
                   ) -> MillipedeInputData:
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            '''
                Input validation
            '''
            if experiment_merge_strategy == MillipedeExperimentMergeStrategy.SUM:
                assert replicate_merge_strategy == MillipedeReplicateMergeStrategy.SUM, "replicate_merge_strategy must be SUM if experiment_merge_strategy is SUM"
            
            #
            # Process the replicate dataframes:
            # type List[pd.DataFrame] if relicates are combined
            # type List[List[pd.DataFrame]] if replicates are separate
            #
            merged_experiment_df_list: List[List[pd.DataFrame]] = []
            # Iterate through the experiments
            for experiment_index, unprocessed_exp_merged_rep_df_list in enumerate(millipede_input_data_loader.unprocessed_merged_experiment_df_list):
                exp_merged_rep_df_list: List[pd.DataFrame] = []
                for _, unprocessed_merged_exp_rep_df in enumerate(unprocessed_exp_merged_rep_df_list):
                    # filter based on the per_replicate_each_condition_num_cutoff
                    merged_exp_rep_df = unprocessed_merged_exp_rep_df.copy()
                    nucleotide_ids = [col for col in merged_exp_rep_df.columns if ">" in col]

                    merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df[millipede_input_data_loader.baseline_pop_df_reads_colname] >= cutoff_specification.per_replicate_each_condition_num_cutoff]
                    merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df[millipede_input_data_loader.enriched_pop_df_reads_colname] >= cutoff_specification.per_replicate_each_condition_num_cutoff]
                    if millipede_input_data_loader.presort_pop_fn_experiment_list is not None:
                        merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df[millipede_input_data_loader.presort_pop_df_reads_colname] >= cutoff_specification.per_replicate_presort_condition_num_cutoff]
                    merged_exp_rep_df['total_reads'] = merged_exp_rep_df[millipede_input_data_loader.baseline_pop_df_reads_colname] + merged_exp_rep_df[millipede_input_data_loader.enriched_pop_df_reads_colname]

                    # filter on total reads based on the per_replicate_all_condition_num_cutoff
                    total_alleles_pre_filter = merged_exp_rep_df.values.shape[0]
                    merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df["total_reads"] >= cutoff_specification.per_replicate_all_condition_num_cutoff]
                    
                    # Add to the replicate list
                    exp_merged_rep_df_list.append(merged_exp_rep_df)



                # TODO: Perform normalization after per-condition and all-condition filtering
                '''
                    Per-condition filtering
                ''' 
                # Add a temporary replicate_id to columns for easy de-concatenation
                for rep_i, merged_exp_rep_df in enumerate(exp_merged_rep_df_list):
                    merged_exp_rep_df["rep_i"] = rep_i

                # Concatenate the replicates together to then perform a groupby across all replicates for filtering
                merged_exp_reps_df: pd.DataFrame = pd.concat(exp_merged_rep_df_list)    
                
                # Per-condition filtering
                per_condition_reads_filter = lambda reads_colname, reads_num_cutoff, acceptable_rep_count, df: sum(df[reads_colname] >= reads_num_cutoff) >= acceptable_rep_count 
                if (cutoff_specification.baseline_pop_per_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.baseline_pop_per_condition_acceptable_rep_count > 0):
                    print(f"Running baseline per-condition filtering with num_cutoff={cutoff_specification.baseline_pop_per_condition_each_replicate_num_cutoff} and acceptable_rep_count={cutoff_specification.baseline_pop_per_condition_acceptable_rep_count}")
                    merged_exp_reps_df = merged_exp_reps_df.groupby(nucleotide_ids, as_index=False).filter(partial(per_condition_reads_filter(millipede_input_data_loader.baseline_pop_df_reads_colname, cutoff_specification.baseline_pop_per_condition_each_replicate_num_cutoff, cutoff_specification.baseline_pop_per_condition_acceptable_rep_count)))
                if (cutoff_specification.enriched_pop_per_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.enriched_pop_per_condition_acceptable_rep_count > 0):
                    print(f"Running enriched per-condition filtering with num_cutoff={cutoff_specification.enriched_pop_per_condition_each_replicate_num_cutoff} and acceptable_rep_count={cutoff_specification.enriched_pop_per_condition_acceptable_rep_count}")
                    merged_exp_reps_df = merged_exp_reps_df.groupby(nucleotide_ids, as_index=False).filter(partial(per_condition_reads_filter(millipede_input_data_loader.enriched_pop_df_reads_colname, cutoff_specification.enriched_pop_per_condition_each_replicate_num_cutoff, cutoff_specification.enriched_pop_per_condition_acceptable_rep_count)))
                if (cutoff_specification.presort_pop_per_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.presort_pop_per_condition_acceptable_rep_count > 0):
                    print(f"Running enriched per-condition filtering with num_cutoff={cutoff_specification.presort_pop_per_condition_each_replicate_num_cutoff} and acceptable_rep_count={cutoff_specification.presort_pop_per_condition_acceptable_rep_count}")
                    merged_exp_reps_df = merged_exp_reps_df.groupby(nucleotide_ids, as_index=False).filter(partial(per_condition_reads_filter(millipede_input_data_loader.presort_pop_df_reads_colname, cutoff_specification.presort_pop_per_condition_each_replicate_num_cutoff, cutoff_specification.presort_pop_per_condition_acceptable_rep_count)))


                # All-condition filtering
                if ((cutoff_specification.baseline_pop_all_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.baseline_pop_all_condition_acceptable_rep_count > 0)) | ((cutoff_specification.enriched_pop_all_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.enriched_pop_all_condition_acceptable_rep_count > 0)) | ((cutoff_specification.presort_pop_all_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.presort_pop_all_condition_acceptable_rep_count > 0)) :
                    print(f"Running all-condition filtering with enriched_num_cutoff={cutoff_specification.enriched_pop_all_condition_each_replicate_num_cutoff}, enriched_acceptable_rep_count={cutoff_specification.enriched_pop_all_condition_acceptable_rep_count}, baseline_num_cutoff={cutoff_specification.baseline_pop_all_condition_each_replicate_num_cutoff}, baseline_acceptable_rep_count={cutoff_specification.baseline_pop_all_condition_acceptable_rep_count}, presort_num_cutoff={cutoff_specification.presort_pop_all_condition_each_replicate_num_cutoff}, presort_acceptable_rep_count={cutoff_specification.presort_pop_all_condition_acceptable_rep_count}")

                    def all_condition_filter_func(df: pd.DataFrame):
                        return per_condition_reads_filter(millipede_input_data_loader.baseline_pop_df_reads_colname, cutoff_specification.baseline_pop_all_condition_each_replicate_num_cutoff, cutoff_specification.baseline_pop_all_condition_acceptable_rep_count, df) | per_condition_reads_filter(millipede_input_data_loader.enriched_pop_df_reads_colname, cutoff_specification.enriched_pop_all_condition_each_replicate_num_cutoff, cutoff_specification.enriched_pop_all_condition_acceptable_rep_count, df) | per_condition_reads_filter(millipede_input_data_loader.presort_pop_df_reads_colname, cutoff_specification.presort_pop_all_condition_each_replicate_num_cutoff, cutoff_specification.presort_pop_all_condition_acceptable_rep_count, df)
                    
                    merged_exp_reps_df = merged_exp_reps_df.groupby(nucleotide_ids, as_index=False).filter(all_condition_filter_func)

                # De-concatenate back into separate replicate by groupby on temporary rep_i column
                exp_merged_rep_df_list = [merged_exp_rep_df for _, merged_exp_rep_df in merged_exp_reps_df.groupby("rep_i")]

                '''
                    Perform normalization after filtering
                '''
                def normalize_func(merged_exp_rep_df):
                    merged_exp_rep_normalized_df: pd.DataFrame = normalize_counts(merged_exp_rep_df, millipede_input_data_loader.enriched_pop_df_reads_colname, millipede_input_data_loader.baseline_pop_df_reads_colname, nucleotide_ids, design_matrix_processing_specification.wt_normalization, design_matrix_processing_specification.total_normalization, millipede_input_data_loader.presort_pop_df_reads_colname) 
                    return merged_exp_rep_normalized_df
                # TODO 20240808 Can implement normalization that requires all replicates "exp_merged_rep_df_list" ie size factors from Zain
                exp_merged_rep_df_list = [normalize_func(merged_exp_rep_df) for merged_exp_rep_df in exp_merged_rep_df_list]
                

                '''
                    Handle all replicates depending on provided strategy
                '''
                # If replicate_merge_strategy is SUM, sum the replicates together 
                if replicate_merge_strategy == MillipedeReplicateMergeStrategy.SUM:
                    nucleotide_ids = [col for col in exp_merged_rep_df_list[0].columns if ">" in col]

                    merged_exp_reps_df: pd.DataFrame = pd.concat(exp_merged_rep_df_list).groupby(nucleotide_ids, as_index=False).sum() 
                    merged_exp_reps_df: pd.DataFrame = merged_exp_reps_df[merged_exp_reps_df["total_reads"] >= cutoff_specification.all_replicate_num_cutoff] # Filter
                    merged_experiment_df_list.append(merged_exp_reps_df)


                elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.COVARIATE:
                    # DEVELOPER NOTE: Ensure that intercept_postfix between per-replicate and per-experiment are different
                    merged_exp_reps_df: pd.DataFrame = pd.concat([self.__get_intercept_df(exp_merged_rep_df_list, experiment_id=experiment_index), pd.concat(exp_merged_rep_df_list, ignore_index=True)], axis=1)
                    merged_experiment_df_list.append(merged_exp_reps_df)


                elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.SEPARATE:
                    merged_experiment_df_list.append(exp_merged_rep_df_list)

                # NOTE 6/13/2024: Decided note to implement score as PYDeseq2 score, low priority
                #elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.PYDEQ:
                #    merged_exp_reps_df: pd.DataFrame = run_pydeseq2(exp_merged_rep_df_list)
                #    merged_experiment_df_list.append(merged_exp_reps_df)
                
                elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.MODELLED_COMBINED:
                    # TODO: Perform error handling. Double check that each dataframe actually has a WT column
                    # This gets the WT allele from each replicate, as this will be used as the negative for CRISPR-Shrinkage
                    # Set negative counts
                    wt_allele_rep_df = [merged_rep_df[merged_rep_df[nucleotide_ids].sum(axis=1) == 0] for merged_rep_df in exp_merged_rep_df_list]
                    
                    # Rename the dataframe to differentiate counts between reps
                    wt_allele_rep_df_renamed = []
                    for rep_i, df in enumerate(wt_allele_rep_df):
                        df = df.rename(columns={millipede_input_data_loader.enriched_pop_df_reads_colname: millipede_input_data_loader.enriched_pop_df_reads_colname+"_rep{}".format(rep_i), millipede_input_data_loader.baseline_pop_df_reads_colname: millipede_input_data_loader.baseline_pop_df_reads_colname+"_rep{}".format(rep_i)})
                        wt_allele_rep_df_renamed.append(df)
                    
                    # Group by allele
                    nucleotide_ids = [col for col in wt_allele_rep_df_renamed[0].columns if ">" in col]
                    wt_allele_rep_df_merged = pd.concat(wt_allele_rep_df).groupby(nucleotide_ids, as_index=False).sum() # This is for the final dataframe
                    wt_allele_rep_df_renamed_merged = pd.concat(wt_allele_rep_df_renamed).groupby(nucleotide_ids, as_index=False)
                    
                    
                    
                    negative_guides = []
                    for index, (name, group) in enumerate(wt_allele_rep_df_renamed_merged):
                        group_noNaN = group.fillna(0)
                        sample_population_raw_count_reps_observation = np.asarray([group_noNaN[millipede_input_data_loader.enriched_pop_df_reads_colname+"_rep{}".format(rep_i)].sum() for rep_i in millipede_input_data_loader.reps])
                        control_population_raw_count_reps_observation = np.asarray([group_noNaN[millipede_input_data_loader.baseline_pop_df_reads_colname+"_rep{}".format(rep_i)].sum() for rep_i in millipede_input_data_loader.reps])
                        # TODO: Later can add more info to guide, i.e. the allele. But setting the identifer as the df index is good and possibly sufficient.
                        guide = crispr_shrinkage.Guide(identifier="negative_{}".format(index), position=None, sample_population_raw_count_reps=sample_population_raw_count_reps_observation, control_population_raw_count_reps=control_population_raw_count_reps_observation, is_explanatory=True)
                        negative_guides.append(guide)
                        
                    
                    
                    # Get alleles that are mutated
                    mut_allele_rep_df = [merged_rep_df[merged_rep_df[nucleotide_ids].sum(axis=1) > 0] for merged_rep_df in exp_merged_rep_df_list]
                    
                    # Rename the dataframe to differentiate counts between reps
                    mut_allele_rep_df_renamed = []
                    for rep_i, df in enumerate(mut_allele_rep_df):
                        df = df.rename(columns={millipede_input_data_loader.enriched_pop_df_reads_colname: millipede_input_data_loader.enriched_pop_df_reads_colname+"_rep{}".format(rep_i), millipede_input_data_loader.baseline_pop_df_reads_colname: millipede_input_data_loader.baseline_pop_df_reads_colname+"_rep{}".format(rep_i)})
                        mut_allele_rep_df_renamed.append(df)
                    
                    # Group by allele
                    nucleotide_ids = [col for col in mut_allele_rep_df_renamed[0].columns if ">" in col]
                    mut_allele_rep_df_merged = pd.concat(mut_allele_rep_df).groupby(nucleotide_ids, as_index=False).sum()
                    mut_allele_rep_df_renamed_merged = pd.concat(mut_allele_rep_df_renamed).groupby(nucleotide_ids, as_index=False)

                    # Get counts of each replicate for each allele. In CRISPR-Shrinkage, each allele will be treated as a guide entity 
                    observation_guides = []
                    for index, (name, group) in enumerate(mut_allele_rep_df_renamed_merged):
                        group_noNaN = group.fillna(0)
                        sample_population_raw_count_reps_observation = np.asarray([group_noNaN[millipede_input_data_loader.enriched_pop_df_reads_colname+"_rep{}".format(rep_i)].sum() for rep_i in millipede_input_data_loader.reps])
                        control_population_raw_count_reps_observation = np.asarray([group_noNaN[millipede_input_data_loader.baseline_pop_df_reads_colname+"_rep{}".format(rep_i)].sum() for rep_i in millipede_input_data_loader.reps])
                        # TODO: Later can add more info to guide, i.e. the allele. But setting the identifer as the df index is good and possibly sufficient.
                        guide = crispr_shrinkage.Guide(identifier="observation_{}".format(index), position=None, sample_population_raw_count_reps=sample_population_raw_count_reps_observation, control_population_raw_count_reps=control_population_raw_count_reps_observation, is_explanatory=True)
                        observation_guides.append(guide)
                        
                    shrinkage_results = crispr_shrinkage.perform_adjustment(
                        negative_control_guides = negative_guides,
                        positive_control_guides = [],
                        observation_guides = observation_guides,
                        num_replicates = len(millipede_input_data_loader.reps),
                        include_observational_guides_in_fit = shrinkage_input.include_observational_guides_in_fit,
                        include_positive_control_guides_in_fit = shrinkage_input.include_positive_control_guides_in_fit,
                        sample_population_scaling_factors = shrinkage_input.sample_population_scaling_factors_exp_list[experiment_index],
                        control_population_scaling_factors = shrinkage_input.control_population_scaling_factors_exp_list[experiment_index],
                        monte_carlo_trials = shrinkage_input.monte_carlo_trials,
                        enable_neighborhood_prior =  False,
                        neighborhood_bandwidth = 1,
                        neighborhood_imputation_prior_strength = None,
                        neighborhood_imputation_likelihood_strength = None,
                        singleton_imputation_prior_strength = [0.006, 0.006, 0.006],#shrinkage_input.singleton_imputation_prior_strength, TODO 5/15/23: should this be uncommented?
                        deviation_weights = shrinkage_input.deviation_weights,
                        KL_guide_set_weights = shrinkage_input.KL_guide_set_weights,
                        shrinkage_prior_strength = [0, 0, 0],#shrinkage_input.shrinkage_prior_strength, TODO 5/15/23: should this be uncommented?
                        posterior_estimator = shrinkage_input.posterior_estimator,
                        random_seed = shrinkage_input.random_seed,
                        cores=shrinkage_input.cores,
                        neighborhood_optimization_guide_sample_size = None
                        )
                    
                        
                    wt_allele_rep_df_merged_updated = wt_allele_rep_df_merged.copy()

                    wt_allele_rep_df_merged_updated.loc[0,"score"] = shrinkage_results.adjusted_negative_control_guides[0].LFC_estimate_combined_rescaled
                    wt_allele_rep_df_merged_updated.loc[0,"scale_factor"] = shrinkage_results.adjusted_negative_control_guides[0].LFC_estimate_combined_std_rescaled/5 + 0.0001

                    mut_allele_rep_df_merged_updated = mut_allele_rep_df_merged.copy()
                    mut_allele_rep_df_merged_updated["score"] = [observation_guide.LFC_estimate_combined_rescaled for observation_guide in shrinkage_results.adjusted_observation_guides]
                    mut_allele_rep_df_merged_updated["scale_factor"] = [observation_guide.LFC_estimate_combined_std_rescaled/5 for observation_guide in shrinkage_results.adjusted_observation_guides]

                    merged_exp_reps_df = pd.concat([wt_allele_rep_df_merged_updated, mut_allele_rep_df_merged_updated], axis=0)
                    
                    merged_experiment_df_list.append(merged_exp_reps_df)
                        
                # TODO: Any way to make more modular?
                elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.MODELLED_SEPARATE:
                    # TODO: Perform error handling. Double check that each dataframe actually has a WT column
                    # This gets the WT allele from each replicate, as this will be used as the negative for CRISPR-Shrinkage
                    # Set negative counts
                    merged_rep_df_list_updated = []
                    for rep_i in millipede_input_data_loader.reps:
                        merged_exp_rep_df = exp_merged_rep_df_list[rep_i]
                        wt_allele_df = merged_exp_rep_df[merged_exp_rep_df[nucleotide_ids].sum(axis=1) == 0]

                        # Rename the dataframe to differentiate counts between reps
                        wt_allele_df_renamed = wt_allele_df.rename(columns={millipede_input_data_loader.enriched_pop_df_reads_colname: millipede_input_data_loader.enriched_pop_df_reads_colname+"_rep{}".format(rep_i), millipede_input_data_loader.baseline_pop_df_reads_colname: millipede_input_data_loader.baseline_pop_df_reads_colname+"_rep{}".format(rep_i)})

                        # Group by allele
                        nucleotide_ids = [col for col in wt_allele_df_renamed.columns if ">" in col]
                        wt_allele_df_merged = wt_allele_df.groupby(nucleotide_ids, as_index=False).sum() # This is for the final dataframe
                        wt_allele_df_renamed_merged = wt_allele_df_renamed.groupby(nucleotide_ids, as_index=False)



                        negative_guides = []
                        for index, (name, group) in enumerate(wt_allele_df_renamed_merged):
                            group_noNaN = group.fillna(0)
                            sample_population_raw_count_reps_observation = np.asarray([group_noNaN[millipede_input_data_loader.enriched_pop_df_reads_colname+"_rep{}".format(rep_i)].sum()])
                            control_population_raw_count_reps_observation = np.asarray([group_noNaN[millipede_input_data_loader.baseline_pop_df_reads_colname+"_rep{}".format(rep_i)].sum()])
                            # TODO: Later can add more info to guide, i.e. the allele. But setting the identifer as the df index is good and possibly sufficient.
                            guide = crispr_shrinkage.Guide(identifier="negative_{}".format(index), position=None, sample_population_raw_count_reps=sample_population_raw_count_reps_observation, control_population_raw_count_reps=control_population_raw_count_reps_observation, is_explanatory=True)
                            negative_guides.append(guide)



                        # Get alleles that are mutated
                        mut_allele_df = merged_exp_rep_df[merged_exp_rep_df[nucleotide_ids].sum(axis=1) > 0]

                        # Rename the dataframe to differentiate counts between reps
                        mut_allele_df_renamed = df.rename(columns={millipede_input_data_loader.enriched_pop_df_reads_colname: millipede_input_data_loader.enriched_pop_df_reads_colname+"_rep{}".format(rep_i), millipede_input_data_loader.baseline_pop_df_reads_colname: millipede_input_data_loader.baseline_pop_df_reads_colname+"_rep{}".format(rep_i)})

                        # Group by allele
                        nucleotide_ids = [col for col in mut_allele_df_renamed.columns if ">" in col]
                        mut_allele_df_merged = mut_allele_df.groupby(nucleotide_ids, as_index=False).sum()
                        mut_allele_df_renamed_merged = mut_allele_df_renamed.groupby(nucleotide_ids, as_index=False)

                        # Get counts of each replicate for each allele. In CRISPR-Shrinkage, each allele will be treated as a guide entity 
                        observation_guides = []
                        for index, (name, group) in enumerate(mut_allele_df_renamed_merged):
                            group_noNaN = group.fillna(0)
                            sample_population_raw_count_reps_observation = np.asarray([group_noNaN[millipede_input_data_loader.enriched_pop_df_reads_colname+"_rep{}".format(rep_i)].sum()])
                            control_population_raw_count_reps_observation = np.asarray([group_noNaN[millipede_input_data_loader.baseline_pop_df_reads_colname+"_rep{}".format(rep_i)].sum()])
                            # TODO: Later can add more info to guide, i.e. the allele. But setting the identifer as the df index is good and possibly sufficient.
                            guide = crispr_shrinkage.Guide(identifier="observation_{}".format(index), position=None, sample_population_raw_count_reps=sample_population_raw_count_reps_observation, control_population_raw_count_reps=control_population_raw_count_reps_observation, is_explanatory=True)
                            observation_guides.append(guide)

                        shrinkage_results = crispr_shrinkage.perform_adjustment(
                            negative_control_guides = negative_guides,
                            positive_control_guides = [],
                            observation_guides = observation_guides,
                            num_replicates = 1,
                            include_observational_guides_in_fit = shrinkage_input.include_observational_guides_in_fit,
                            include_positive_control_guides_in_fit = shrinkage_input.include_positive_control_guides_in_fit,
                            sample_population_scaling_factors = shrinkage_input.sample_population_scaling_factors_exp_list[experiment_index],
                            control_population_scaling_factors = shrinkage_input.control_population_scaling_factors_exp_list[experiment_index],
                            monte_carlo_trials = shrinkage_input.monte_carlo_trials,
                            enable_neighborhood_prior =  False,
                            neighborhood_bandwidth = 1,
                            neighborhood_imputation_prior_strength = None,
                            neighborhood_imputation_likelihood_strength = None,
                            singleton_imputation_prior_strength = [0.006, 0.006, 0.006],#shrinkage_input.singleton_imputation_prior_strength,
                            deviation_weights = shrinkage_input.deviation_weights,
                            KL_guide_set_weights = shrinkage_input.KL_guide_set_weights,
                            shrinkage_prior_strength = [0, 0, 0],#shrinkage_input.shrinkage_prior_strength, 
                            posterior_estimator = shrinkage_input.posterior_estimator,
                            random_seed = shrinkage_input.random_seed,
                            cores=shrinkage_input.cores,
                            neighborhood_optimization_guide_sample_size = None
                            )


                        wt_allele_df_merged_updated = wt_allele_df_merged.copy()

                        wt_allele_df_merged_updated.loc[0,"score"] = shrinkage_results.adjusted_negative_control_guides[0].LFC_estimate_combined_rescaled 
                        wt_allele_df_merged_updated.loc[0,"scale_factor"] = shrinkage_results.adjusted_negative_control_guides[0].LFC_estimate_combined_std_rescaled/2 + 0.0001

                        mut_allele_df_merged_updated = mut_allele_df_merged.copy()
                        mut_allele_df_merged_updated["score"] = [observation_guide.LFC_estimate_combined_rescaled for observation_guide in shrinkage_results.adjusted_observation_guides]
                        mut_allele_df_merged_updated["scale_factor"] = [observation_guide.LFC_estimate_combined_std_rescaled/2 for observation_guide in shrinkage_results.adjusted_observation_guides]

                        merged_exp_reps_df = pd.concat([wt_allele_rep_df_merged_updated, mut_allele_rep_df_merged_updated], axis=0)
                        
                        merged_rep_df_list_updated.append(merged_exp_reps_df)
                    merged_experiment_df_list.append(merged_rep_df_list_updated)
                else:
                    raise Exception("Developer error: Unexpected value for MillipedeReplicateMergeStrategy: {}".format(replicate_merge_strategy))
            
            '''
                Handle all experiments depending on provided strategy
            '''
            __add_supporting_columns_partial = partial(self.__add_supporting_columns,
                                                       enriched_pop_df_reads_colname=millipede_input_data_loader.enriched_pop_df_reads_colname,                               
                                                       baseline_pop_df_reads_colname= millipede_input_data_loader.baseline_pop_df_reads_colname,
                                                       presort_pop_df_reads_colname=millipede_input_data_loader.presort_pop_df_reads_colname,
                                                       bounded_score=design_matrix_processing_specification.bounded_score,
                                                       score_psuedocount=design_matrix_processing_specification.score_psuedocount,
                                                       sigma_scale_normalized= design_matrix_processing_specification.sigma_scale_normalized,
                                                       decay_sigma_scale= design_matrix_processing_specification.decay_sigma_scale,
                                                       use_2d_decay_function=design_matrix_processing_specification.use_2d_decay_function,
                                                       
                                                       K_enriched=design_matrix_processing_specification.K_enriched, 
                                                       K_baseline=design_matrix_processing_specification.K_baseline, 
                                                       a_parameter_enriched=design_matrix_processing_specification.a_parameter_enriched,
                                                       a_parameter_baseline=design_matrix_processing_specification.a_parameter_baseline,
                                                       c_parameter_enriched=design_matrix_processing_specification.c_parameter_enriched,
                                                       c_parameter_baseline=design_matrix_processing_specification.c_parameter_baseline,
                                                       
                                                       A1_parameter_2D=design_matrix_processing_specification.A1_parameter_2D,
                                                       k1_parameter_enriched_2D=design_matrix_processing_specification.k1_parameter_enriched_2D,
                                                       k1_parameter_baseline_2D=design_matrix_processing_specification.k1_parameter_baseline_2D,
                                                       A2_parameter_2D=design_matrix_processing_specification.A2_parameter_2D,
                                                       k2_parameter_enriched_2D=design_matrix_processing_specification.k2_parameter_enriched_2D,
                                                       k2_parameter_baseline_2D=design_matrix_processing_specification.k2_parameter_baseline_2D,
                                                       C_parameter_2D=design_matrix_processing_specification.C_parameter_2D,


                                                       set_offset_as_default=design_matrix_processing_specification.set_offset_as_default,
                                                       set_offset_as_total_reads=design_matrix_processing_specification.set_offset_as_total_reads,
                                                       set_offset_as_enriched=design_matrix_processing_specification.set_offset_as_enriched,
                                                       set_offset_as_baseline=design_matrix_processing_specification.set_offset_as_baseline,
                                                       set_offset_as_presort=design_matrix_processing_specification.set_offset_as_presort,
                                                       offset_normalized=design_matrix_processing_specification.offset_normalized,
                                                       offset_psuedocount=design_matrix_processing_specification.offset_psuedocount
                                                      )
            


            data = None
            if experiment_merge_strategy == MillipedeExperimentMergeStrategy.SUM:
                nucleotide_ids = [col for col in merged_experiment_df_list[0].columns if ">" in col]
                merged_experiments_df: pd.DataFrame
                merged_experiments_df = pd.concat(merged_experiment_df_list).groupby(nucleotide_ids, as_index=False).sum()
                # Filter rows based on cutoffs
                merged_experiments_df = merged_experiments_df[merged_experiments_df["total_reads"] >= cutoff_specification.all_experiment_num_cutoff]
                #merged_experiments_df = merged_experiments_df[merged_experiments_df["total_reads"] > 0] # Ensure non-zero reads to prevent error during modelling

                merged_experiments_df = __add_supporting_columns_partial(encoding_df = merged_experiments_df)

                if self.remove_empty_features:
                    merged_experiments_df = merged_experiments_df.drop(
                        columns=[col for col in nucleotide_ids if merged_experiments_df[col].sum() == 0]
                    )
                data = merged_experiments_df
            elif experiment_merge_strategy == MillipedeExperimentMergeStrategy.COVARIATE:
                # DEVELOPER NOTE: Ensure that intercept_postfix between per-replicate and per-experiment are different, else there could be overwriting during intercept assignment
                if replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SEPARATE, MillipedeReplicateMergeStrategy.MODELLED_SEPARATE]: # SINGLE MATRIX PER REPLICATE
                    merged_experiment_df_list: List[List[pd.DataFrame]]
                    merged_experiments_df: List[pd.DataFrame]
                    merged_experiments_df = [pd.concat([self.__get_intercept_df(merged_experiment_df_list), pd.concat(merged_experiment_df_i, ignore_index=True)], axis=1) for merged_experiment_df_i in merged_experiment_df_list]
                    merged_experiments_df = [merged_experiments_df_i.fillna(0.0) for merged_experiments_df_i in merged_experiments_df] # TODO 20221021: This is to ensure all intercept values are assigned (since NaNs exist with covariate by experiment) - there is possible if there are other NaN among features that it will be set to 0 unintentionally
                    merged_experiments_df = [__add_supporting_columns_partial(encoding_df = merged_experiments_df_i, replicate_i=replicate_i) for replicate_i, merged_experiments_df_i in enumerate(merged_experiments_df)]
                    #merged_experiments_df = [merged_experiments_df_i[merged_experiments_df_i["total_reads"] > 0] for merged_experiments_df_i in merged_experiments_df] # Ensure non-zero reads to prevent error during modelling
                    
                    if self.remove_empty_features:
                        nucleotide_ids = [col for col in merged_experiments_df.columns if ">" in col]
                        merged_experiments_df = merged_experiments_df.drop(
                            columns=[col for col in nucleotide_ids if merged_experiments_df[col].sum() == 0]
                        )
                        
                    data = merged_experiments_df
                elif replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SUM, MillipedeReplicateMergeStrategy.COVARIATE, MillipedeReplicateMergeStrategy.MODELLED_COMBINED]: # SINGLE MATRIX FOR ALL REPLICATES
                    merged_experiment_df_list: List[pd.DataFrame]
                    merged_experiments_df: pd.DataFrame
                    merged_experiments_df = pd.concat([self.__get_intercept_df(merged_experiment_df_list), pd.concat(merged_experiment_df_list, ignore_index=True)], axis=1)
                    merged_experiments_df = merged_experiments_df.fillna(0.0) # TODO 20221021: This is to ensure all intercept values are assigned (since NaNs exist with covariate by experiment) - there is possible if there are other NaN among features that it will be set to 0 unintentionally
                    merged_experiments_df = __add_supporting_columns_partial(encoding_df = merged_experiments_df)
                    #merged_experiments_df = merged_experiments_df[merged_experiments_df["total_reads"] > 0] # Ensure non-zero reads to prevent error during modelling

                    if self.remove_empty_features:
                        nucleotide_ids = [col for col in merged_experiments_df.columns if ">" in col]
                        merged_experiments_df = merged_experiments_df.drop(
                            columns=[col for col in nucleotide_ids if merged_experiments_df[col].sum() == 0]
                        )

                    data = merged_experiments_df
            elif experiment_merge_strategy == MillipedeExperimentMergeStrategy.SEPARATE:
                if replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SEPARATE, MillipedeReplicateMergeStrategy.MODELLED_SEPARATE]:
                    merged_experiment_df_list: List[List[pd.DataFrame]]
                    merged_experiment_df_list = [[__add_supporting_columns_partial(encoding_df = merged_rep_df, experiment_i=experiment_i, replicate_i=replicate_i) for replicate_i, merged_rep_df in enumerate(merged_rep_df_list)] for experiment_i, merged_rep_df_list in enumerate(merged_experiment_df_list)]
                    #merged_experiment_df_list = [[merged_rep_df[merged_rep_df["total_reads"] > 0] for merged_rep_df in merged_rep_df_list] for merged_rep_df_list in merged_experiment_df_list] # Ensure non-zero reads to prevent error during modelling

                    if self.remove_empty_features:
                        merged_experiment_df_list_tmp = []
                        for merged_experiments_df_list_inner in merged_experiment_df_list:
                            merged_experiment_df_list_inner_tmp = []
                            for merged_experiments_df in merged_experiments_df_list_inner:
                                nucleotide_ids = [col for col in merged_experiments_df.columns if ">" in col]
                                merged_experiments_df_tmp = merged_experiments_df.drop(
                                    columns=[col for col in nucleotide_ids if merged_experiments_df[col].sum() == 0]
                                )
                                merged_experiment_df_list_inner_tmp.append(merged_experiments_df_tmp)
                            merged_experiment_df_list_tmp.append(merged_experiment_df_list_inner_tmp)
                        merged_experiment_df_list=merged_experiment_df_list_tmp

                    data = merged_experiment_df_list
                elif replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SUM, MillipedeReplicateMergeStrategy.COVARIATE, MillipedeReplicateMergeStrategy.MODELLED_COMBINED]:
                    merged_experiment_df_list: List[pd.DataFrame]
                    merged_experiment_df_list = [__add_supporting_columns_partial(encoding_df = merged_reps_df, experiment_i=experiment_i) for experiment_i, merged_reps_df in enumerate(merged_experiment_df_list)]
                    #merged_experiment_df_list = [merged_reps_df[merged_reps_df["total_reads"] > 0] for merged_reps_df in merged_experiment_df_list]
                    
                    if self.remove_empty_features:
                        merged_experiment_df_list_tmp = []
                        for merged_experiments_df in merged_experiment_df_list:
                            nucleotide_ids = [col for col in merged_experiments_df.columns if ">" in col]
                            merged_experiments_df_tmp = merged_experiments_df.drop(
                                columns=[col for col in nucleotide_ids if merged_experiments_df[col].sum() == 0]
                            )
                            merged_experiment_df_list_tmp.append(merged_experiments_df_tmp)
                        merged_experiment_df_list=merged_experiment_df_list_tmp
                        
                    data = merged_experiment_df_list
            else:
                raise Exception("Developer error: Unexpected value for MillipedeExperimentMergeStrategy: {}".format(experiment_merge_strategy))

            millipede_input_data: MillipedeInputData = MillipedeInputData(
                data=data,
                enriched_pop_df_reads_colname=millipede_input_data_loader.enriched_pop_df_reads_colname,
                baseline_pop_df_reads_colname=millipede_input_data_loader.baseline_pop_df_reads_colname,
                presort_pop_df_reads_colname=millipede_input_data_loader.presort_pop_df_reads_colname,
                reps=millipede_input_data_loader.reps,
                replicate_merge_strategy=replicate_merge_strategy, 
                experiment_merge_strategy=experiment_merge_strategy,
                cutoff_specification=cutoff_specification,
                design_matrix_processing_specification=design_matrix_processing_specification
            )
            
            return millipede_input_data

    def __determine_full_design_matrix_set(self, millipede_model_specification_set: Mapping[str, MillipedeModelSpecification]) -> Mapping[Tuple[MillipedeReplicateMergeStrategy, MillipedeExperimentMergeStrategy, MillipedeCutoffSpecification, MillipedeShrinkageInput], List[str]]:
        """
            This determines what set of design matrices to generate based on the set of Millipede model specifications - this is determined based on the replicate/experiment merge strategies

            The returned variable is a dictionary with the replicate/experiment merge strategy tuple as the key and the list of Millipede model specifications IDs as the value to ensure the model specification that each design matrix maps to.
        """
        millipede_design_matrix_set: Mapping[Tuple[MillipedeReplicateMergeStrategy, MillipedeExperimentMergeStrategy, MillipedeCutoffSpecification, MillipedeShrinkageInput, MillipedeDesignMatrixProcessingSpecification], List[str]] = defaultdict(list)

        millipede_model_specification_id: str
        millipede_model_specification: MillipedeModelSpecification
        for millipede_model_specification_id, millipede_model_specification in millipede_model_specification_set.items():
            millipede_design_matrix_set[(millipede_model_specification.replicate_merge_strategy, millipede_model_specification.experiment_merge_strategy, millipede_model_specification.cutoff_specification, millipede_model_specification.shrinkage_input, millipede_model_specification.design_matrix_processing_specification)].append(millipede_model_specification_id)

        return millipede_design_matrix_set

    def __add_supporting_columns(self, 
                                 encoding_df: pd.DataFrame, 
                                 enriched_pop_df_reads_colname: str, 
                                 baseline_pop_df_reads_colname: str,
                                 presort_pop_df_reads_colname: Optional[str],
                                 bounded_score:bool,
                                 score_psuedocount:float,
                                 sigma_scale_normalized: bool,
                                 decay_sigma_scale: bool,
                                 use_2d_decay_function: bool,
                                 
                                 set_offset_as_default: bool,
                                 set_offset_as_total_reads: bool,
                                 set_offset_as_enriched: bool,
                                 set_offset_as_baseline: bool,
                                 set_offset_as_presort: bool,
                                 offset_normalized: bool,
                                 offset_psuedocount: int,

                                 K_enriched: Union[float, List[float], List[List[float]]] = None,
                                 K_baseline: Union[float, List[float], List[List[float]]] = None,
                                 a_parameter_enriched: Union[float, List[float], List[List[float]]] = None,
                                 a_parameter_baseline: Union[float, List[float], List[List[float]]] = None,
                                 c_parameter_enriched: Union[float, List[float], List[List[float]]] = None,
                                 c_parameter_baseline: Union[float, List[float], List[List[float]]] = None,

                                 A1_parameter_2D: Union[float, List[float], List[List[float]]] = None,
                                 k1_parameter_enriched_2D: Union[float, List[float], List[List[float]]] = None,
                                 k1_parameter_baseline_2D: Union[float, List[float], List[List[float]]] = None,

                                 A2_parameter_2D: Union[float, List[float], List[List[float]]] = None,
                                 k2_parameter_enriched_2D: Union[float, List[float], List[List[float]]] = None,
                                 k2_parameter_baseline_2D: Union[float, List[float], List[List[float]]] = None,

                                 C_parameter_2D: Union[float, List[float], List[List[float]]] = None,
                                 experiment_i: Optional[int] = None,
                                 replicate_i: Optional[int] = None
                                ) -> pd.DataFrame:
        # construct the simplest possible continuous-valued response variable.
        # this response variable is in [-1, 1]
        
        # Get intercept exp and reps for setting :
        intercept_columns = [col for col in  encoding_df.columns if "intercept" in col]
        intercept_columns_samples = []
        if len(intercept_columns) > 0:
            exp_indices = []
            rep_indices = []
            for col in intercept_columns:
                rep_index = col.find("rep")
                if rep_index != -1:
                    intercept_rep_i = int(col[(rep_index + len("rep")):])
                    rep_indices.append(intercept_rep_i)
                    exp_index = col.find("exp")
                    if exp_index != -1:
                        intercept_exp_i = int(col[(exp_index + len("exp")):rep_index-1])
                        exp_indices.append(intercept_exp_i)
                    else:
                        raise Exception(f"No exp found in intercept column {col}. This is a developer bug, contact developers (see the cripsr-millipeede GitHub page for contact)")
                    intercept_columns_samples.append(col)
                else:
                    # DEPRECATED: If there is exp#, there must be exp#_rep# as well. Therefore we don't want to add duplicate sample encoding DFs
                    pass
                    #exp_index = col.find("exp")
                    #if (exp_index != -1) and (replicate_i is not None):
                    #    intercept_exp_i = int(col[(exp_index + len("exp")):])
                    #    exp_indices.append(intercept_exp_i)
                    #    rep_indices.append(replicate_i) # Add the explit replicate ID if only the experiment ID was found in the intercept column
        

        # Original
        enriched_read_counts = encoding_df[enriched_pop_df_reads_colname]
        baseline_read_counts = encoding_df[baseline_pop_df_reads_colname]
        total_read_counts = encoding_df['total_reads']
        
        if 'score' not in encoding_df.columns: 
            if bounded_score:
                encoding_df['score'] = (enriched_read_counts - baseline_read_counts) / (enriched_read_counts + baseline_read_counts + score_psuedocount) 
            else:
                encoding_df['score'] = np.log2( (enriched_read_counts + score_psuedocount) / (baseline_read_counts + score_psuedocount) )
            encoding_df = encoding_df[~encoding_df['score'].isna()] # Remove rows where score is NA (due to 0 counts)
            
        # create scale_factor for normal likelihood model
        #if 'scale_factor' not in encoding_df.columns: 
            #encoding_df['scale_factor'] = 1.0 / np.sqrt(encoding_df['total_reads']) # NOTE: Intentionally keeping the total_reads as the raw to avoid being impact by normalization - this could be subject to change
        if 'scale_factor' not in encoding_df.columns:
            
            def set_scale_factor(
                input_encoding_df, 
                use_2d_decay_function,
                K_enriched_selected=None, 
                K_baseline_selected=None, 
                a_parameter_enriched_selected=None, 
                a_parameter_baseline_selected=None, 
                c_parameter_enriched_selected=None, 
                c_parameter_baseline_selected=None,

                A1_parameter_2D_selected = None,
                k1_parameter_enriched_2D_selected = None,
                k1_parameter_baseline_2D_selected = None,

                A2_parameter_2D_selected = None,
                k2_parameter_enriched_2D_selected = None,
                k2_parameter_baseline_2D_selected = None,

                C_parameter_2D_selected = None):

                if use_2d_decay_function is True:
                    input_encoding_df["A1_parameter_2D_selected"] = A1_parameter_2D_selected
                    input_encoding_df["k1_parameter_enriched_2D_selected"] = k1_parameter_enriched_2D_selected
                    input_encoding_df["k1_parameter_baseline_2D_selected"] = k1_parameter_baseline_2D_selected
                    
                    input_encoding_df["A2_parameter_2D_selected"] = A2_parameter_2D_selected
                    input_encoding_df["k2_parameter_enriched_2D_selected"] = k2_parameter_enriched_2D_selected
                    input_encoding_df["k2_parameter_baseline_2D_selected"] = k2_parameter_baseline_2D_selected
                    input_encoding_df["C_parameter_2D_selected"] = C_parameter_2D_selected
                else:
                    input_encoding_df["K_enriched"] = K_enriched_selected
                    input_encoding_df["K_baseline"] = K_baseline_selected
                    input_encoding_df["a_parameter_enriched"] = a_parameter_enriched_selected
                    input_encoding_df["a_parameter_baseline"] = a_parameter_baseline_selected
                    input_encoding_df["c_parameter_enriched"] = c_parameter_enriched_selected
                    input_encoding_df["c_parameter_baseline"] = c_parameter_baseline_selected


                if sigma_scale_normalized:
                    if decay_sigma_scale:
                        if use_2d_decay_function is True:
                            input_encoding_df['scale_factor'] = decay_function_2d(
                                                                    enriched_count = input_encoding_df[enriched_pop_df_reads_colname],
                                                                    baseline_count = input_encoding_df[baseline_pop_df_reads_colname],
                                                                    A1_parameter_2D = A1_parameter_2D_selected,
                                                                    k1_parameter_enriched_2D = k1_parameter_enriched_2D_selected,
                                                                    k1_parameter_baseline_2D = k1_parameter_baseline_2D_selected,
                                                                    A2_parameter_2D = A2_parameter_2D_selected,
                                                                    k2_parameter_enriched_2D = k2_parameter_enriched_2D_selected,
                                                                    k2_parameter_baseline_2D = k2_parameter_baseline_2D_selected,
                                                                    C_parameter_2D = C_parameter_2D_selected
                                                                )
                        else:
                            input_encoding_df['scale_factor'] = ((decay_function(input_encoding_df[enriched_pop_df_reads_colname], K_enriched_selected, a_parameter_enriched_selected, c_parameter_enriched_selected))  + (decay_function(input_encoding_df[baseline_pop_df_reads_colname], K_baseline_selected, a_parameter_baseline_selected, c_parameter_baseline_selected)))/2 
                    else:
                        input_encoding_df['scale_factor'] = (K_enriched_selected / np.sqrt(input_encoding_df[enriched_pop_df_reads_colname])) + (input_encoding_df / np.sqrt(input_encoding_df[baseline_pop_df_reads_colname]))
                else:
                    if decay_sigma_scale:
                        if use_2d_decay_function is True:
                            input_encoding_df['scale_factor'] = decay_function_2d(
                                                                    enriched_count = input_encoding_df[enriched_pop_df_reads_colname + "_raw"],
                                                                    baseline_count = input_encoding_df[baseline_pop_df_reads_colname + "_raw"],
                                                                    A1_parameter_2D = A1_parameter_2D_selected,
                                                                    k1_parameter_enriched_2D = k1_parameter_enriched_2D_selected,
                                                                    k1_parameter_baseline_2D = k1_parameter_baseline_2D_selected,
                                                                    A2_parameter_2D = A2_parameter_2D_selected,
                                                                    k2_parameter_enriched_2D = k2_parameter_enriched_2D_selected,
                                                                    k2_parameter_baseline_2D = k2_parameter_baseline_2D_selected,
                                                                    C_parameter_2D = C_parameter_2D_selected
                                                                )
                        else:
                            input_encoding_df['scale_factor'] = ((decay_function(input_encoding_df[enriched_pop_df_reads_colname + "_raw"], K_enriched_selected, a_parameter_enriched_selected, c_parameter_enriched_selected)) + (decay_function(input_encoding_df[baseline_pop_df_reads_colname + "_raw"], K_baseline_selected, a_parameter_baseline_selected, c_parameter_baseline_selected)))/2 
                    else:
                        input_encoding_df['scale_factor'] = (K_enriched_selected / np.sqrt(input_encoding_df[enriched_pop_df_reads_colname + "_raw"])) + (K_baseline_selected / np.sqrt(input_encoding_df[baseline_pop_df_reads_colname + "_raw"]))
                return input_encoding_df


            def retrieve_sample_parameter(parameter_input, experiment_index, replicate_index):
                if parameter_input is None:
                    return None
                if type(parameter_input) is list:
                    assert replicate_index is not None, "Replicate index must be provided"
                    if type(parameter_input[0]) is list:
                        assert experiment_index is not None, "Experiment index must be provided"
                        parameter_input_selected = parameter_input[experiment_index][replicate_index]
                    else:
                        parameter_input_selected = parameter_input[rep_index]
                else:
                    parameter_input_selected = parameter_input
                return parameter_input_selected
    
            # If the encoding as intercept columns, then extract the available exp/rep indices, subset by each exp/rep, get the correct sigma_scale parameters, and update the encoding DF
            if intercept_columns_samples:
                # Iterate through each intercept index to get exp/rep index
                sample_encoding_df_list = []
                for intercept_index, intercept_col in enumerate(intercept_columns_samples):
                    exp_index = exp_indices[intercept_index]
                    rep_index = rep_indices[intercept_index]
                    
                    # Get the corresponding sigma scale parameter based on the exp/rep index
                    K_enriched_selected = retrieve_sample_parameter(K_enriched, experiment_index=exp_index, replicate_index=rep_index)
                    K_baseline_selected = retrieve_sample_parameter(K_baseline, experiment_index=exp_index, replicate_index=rep_index)
                    a_parameter_enriched_selected = retrieve_sample_parameter(a_parameter_enriched, experiment_index=exp_index, replicate_index=rep_index)
                    a_parameter_baseline_selected = retrieve_sample_parameter(a_parameter_baseline, experiment_index=exp_index, replicate_index=rep_index)
                    c_parameter_enriched_selected = retrieve_sample_parameter(c_parameter_enriched, experiment_index=exp_index, replicate_index=rep_index)
                    c_parameter_baseline_selected = retrieve_sample_parameter(c_parameter_baseline, experiment_index=exp_index, replicate_index=rep_index)

                    A1_parameter_2D_selected = retrieve_sample_parameter(A1_parameter_2D, experiment_index=exp_index, replicate_index=rep_index)
                    k1_parameter_enriched_2D_selected = retrieve_sample_parameter(k1_parameter_enriched_2D, experiment_index=exp_index, replicate_index=rep_index)
                    k1_parameter_baseline_2D_selected = retrieve_sample_parameter(k1_parameter_baseline_2D, experiment_index=exp_index, replicate_index=rep_index)
                    A2_parameter_2D_selected = retrieve_sample_parameter(A2_parameter_2D, experiment_index=exp_index, replicate_index=rep_index)
                    k2_parameter_enriched_2D_selected = retrieve_sample_parameter(k2_parameter_enriched_2D, experiment_index=exp_index, replicate_index=rep_index)
                    k2_parameter_baseline_2D_selected = retrieve_sample_parameter(k2_parameter_baseline_2D, experiment_index=exp_index, replicate_index=rep_index)
                    C_parameter_2D_selected = retrieve_sample_parameter(C_parameter_2D, experiment_index=exp_index, replicate_index=rep_index)

                    # Subset the encoding by the intercept index and add scale factor
                    sample_encoding_df = encoding_df[encoding_df[intercept_col] == 1]
                    sample_encoding_df = set_scale_factor(
                        input_encoding_df=sample_encoding_df, 
                        use_2d_decay_function=use_2d_decay_function, 
                        K_enriched_selected=K_enriched_selected, 
                        K_baseline_selected=K_baseline_selected, 
                        a_parameter_enriched_selected=a_parameter_enriched_selected, 
                        a_parameter_baseline_selected=a_parameter_baseline_selected,
                        c_parameter_enriched_selected=c_parameter_enriched_selected, 
                        c_parameter_baseline_selected=c_parameter_baseline_selected, 
                        A1_parameter_2D_selected=A1_parameter_2D_selected,
                        k1_parameter_enriched_2D_selected=k1_parameter_enriched_2D_selected,
                        k1_parameter_baseline_2D_selected=k1_parameter_baseline_2D_selected,
                        A2_parameter_2D_selected=A2_parameter_2D_selected,
                        k2_parameter_enriched_2D_selected=k2_parameter_enriched_2D_selected,
                        k2_parameter_baseline_2D_selected=k2_parameter_baseline_2D_selected,
                        C_parameter_2D_selected=C_parameter_2D_selected)
                    sample_encoding_df_list.append(sample_encoding_df)
                
                # Concatenate all the updated sample encoding DFs into the complete encoding DF
                encoding_df = pd.concat(sample_encoding_df_list, axis=0)
            
            else: # If there are no intercept columns, see if explicit experiment or replicate index is provided
                if replicate_i is None:
                    if experiment_i is None:
                        # If no explicit experiment or replicate index is provided, then expecting a single sigma_scale parameter but must assert the input first
                        assert isinstance(K_enriched, (int, float)), f"K_enriched {K_enriched} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter_enriched, a_parameter_baseline, c_parameter_enriched, c_parameter_baseline) must be an int/float type"
                        assert isinstance(K_baseline, (int, float)), f"K_baseline {K_baseline} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter_enriched, a_parameter_baseline, c_parameter_enriched, c_parameter_baseline) must be an int/float type"
                        assert isinstance(a_parameter_enriched, (int, float)), f"a_parameter {a_parameter_enriched} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter_enriched, a_parameter_baseline, c_parameter_enriched, c_parameter_baseline) must be an int/float type"
                        assert isinstance(a_parameter_baseline, (int, float)), f"a_parameter {a_parameter_baseline} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter_enriched, a_parameter_baseline, c_parameter_enriched, c_parameter_baseline) must be an int/float type"
                        assert isinstance(c_parameter_enriched, (int, float)), f"c_parameter {c_parameter_enriched} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter_enriched, a_parameter_baseline, c_parameter_enriched, c_parameter_baseline) must be an int/float type"
                        assert isinstance(c_parameter_baseline, (int, float)), f"c_parameter {c_parameter_baseline} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter_enriched, a_parameter_baseline, c_parameter_enriched, c_parameter_baseline) must be an int/float type"
                        assert isinstance(A1_parameter_2D, (int, float)) or (A1_parameter_2D is None), f"A1_parameter_2D {A1_parameter_2D} and all 2d_decay_parameters (A1_parameter_2D, k1_parameter_enriched_2D, k1_parameter_baseline_2D, A2_parameter_2D, k2_parameter_enriched_2D, k2_parameter_baseline_2D, C_parameter_2D) must be an int/float type or None"
                        assert isinstance(k1_parameter_enriched_2D, (int, float)) or (k1_parameter_enriched_2D is None), f"k1_parameter_enriched_2D {k1_parameter_enriched_2D} and all 2d_decay_parameters (A1_parameter_2D, k1_parameter_enriched_2D, k1_parameter_baseline_2D, A2_parameter_2D, k2_parameter_enriched_2D, k2_parameter_baseline_2D, C_parameter_2D) must be an int/float type or None"
                        assert isinstance(k1_parameter_baseline_2D, (int, float)) or (k1_parameter_baseline_2D is None), f"k1_parameter_baseline_2D {k1_parameter_baseline_2D} and all 2d_decay_parameters (A1_parameter_2D, k1_parameter_enriched_2D, k1_parameter_baseline_2D, A2_parameter_2D, k2_parameter_enriched_2D, k2_parameter_baseline_2D, C_parameter_2D) must be an int/float type or None"
                        assert isinstance(A2_parameter_2D, (int, float)) or (A2_parameter_2D is None), f"A2_parameter_2D {A2_parameter_2D} and all 2d_decay_parameters (A1_parameter_2D, k1_parameter_enriched_2D, k1_parameter_baseline_2D, A2_parameter_2D, k2_parameter_enriched_2D, k2_parameter_baseline_2D, C_parameter_2D) must be an int/float type or None"
                        assert isinstance(k2_parameter_enriched_2D, (int, float)) or (k2_parameter_enriched_2D is None), f"k2_parameter_enriched_2D {k2_parameter_enriched_2D} and all 2d_decay_parameters (A1_parameter_2D, k1_parameter_enriched_2D, k1_parameter_baseline_2D, A2_parameter_2D, k2_parameter_enriched_2D, k2_parameter_baseline_2D, C_parameter_2D) must be an int/float type or None"
                        assert isinstance(k2_parameter_baseline_2D, (int, float)) or (k2_parameter_baseline_2D is None), f"k2_parameter_baseline_2D {k2_parameter_baseline_2D} and all 2d_decay_parameters (A1_parameter_2D, k1_parameter_enriched_2D, k1_parameter_baseline_2D, A2_parameter_2D, k2_parameter_enriched_2D, k2_parameter_baseline_2D, C_parameter_2D) must be an int/float type or None"
                        assert isinstance(C_parameter_2D, (int, float)) or (C_parameter_2D is None), f"C_parameter_2D {C_parameter_2D} and all 2d_decay_parameters (A1_parameter_2D, k1_parameter_enriched_2D, k1_parameter_baseline_2D, A2_parameter_2D, k2_parameter_enriched_2D, k2_parameter_baseline_2D, C_parameter_2D) must be an int/float type or None"

                        K_enriched_selected = K_enriched
                        K_baseline_selected = K_baseline
                        a_parameter_enriched_selected = a_parameter_enriched
                        a_parameter_baseline_selected = a_parameter_baseline
                        c_parameter_enriched_selected = c_parameter_enriched
                        c_parameter_baseline_selected = c_parameter_baseline

                        A1_parameter_2D_selected = A1_parameter_2D
                        k1_parameter_enriched_2D_selected = k1_parameter_enriched_2D
                        k1_parameter_baseline_2D_selected = k1_parameter_baseline_2D
                        A2_parameter_2D_selected = A2_parameter_2D
                        k2_parameter_enriched_2D_selected = k2_parameter_enriched_2D
                        k2_parameter_baseline_2D_selected = k2_parameter_baseline_2D
                        C_parameter_2D_selected = C_parameter_2D

                else:
                    # If replicate (and experiment) index is provided, get the selected sigma_scale parameters
                    K_enriched_selected = retrieve_sample_parameter(K_enriched, experiment_i, replicate_i)
                    K_baseline_selected = retrieve_sample_parameter(K_baseline, experiment_i, replicate_i)
                    a_parameter_enriched_selected = retrieve_sample_parameter(a_parameter_enriched, experiment_i, replicate_i)
                    a_parameter_baseline_selected = retrieve_sample_parameter(a_parameter_baseline, experiment_i, replicate_i)
                    c_parameter_enriched_selected = retrieve_sample_parameter(c_parameter_enriched, experiment_i, replicate_i)
                    c_parameter_baseline_selected = retrieve_sample_parameter(c_parameter_baseline, experiment_i, replicate_i)

                    A1_parameter_2D_selected = retrieve_sample_parameter(A1_parameter_2D, experiment_i, replicate_i)
                    k1_parameter_enriched_2D_selected = retrieve_sample_parameter(k1_parameter_enriched_2D, experiment_i, replicate_i)
                    k1_parameter_baseline_2D_selected = retrieve_sample_parameter(k1_parameter_baseline_2D, experiment_i, replicate_i)
                    A2_parameter_2D_selected = retrieve_sample_parameter(A2_parameter_2D, experiment_i, replicate_i)
                    k2_parameter_enriched_2D_selected = retrieve_sample_parameter(k2_parameter_enriched_2D, experiment_i, replicate_i)
                    k2_parameter_baseline_2D_selected = retrieve_sample_parameter(k2_parameter_baseline_2D, experiment_i, replicate_i)
                    C_parameter_2D_selected = retrieve_sample_parameter(C_parameter_2D, experiment_i, replicate_i)


                encoding_df = set_scale_factor(
                                input_encoding_df=encoding_df, 
                                use_2d_decay_function=use_2d_decay_function, 
                                K_enriched_selected=K_enriched_selected, 
                                K_baseline_selected=K_baseline_selected, 
                                a_parameter_enriched_selected=a_parameter_enriched_selected, 
                                a_parameter_baseline_selected=a_parameter_baseline_selected,
                                c_parameter_enriched_selected=c_parameter_enriched_selected, 
                                c_parameter_baseline_selected=c_parameter_baseline_selected, 
                                A1_parameter_2D_selected=A1_parameter_2D_selected,
                                k1_parameter_enriched_2D_selected=k1_parameter_enriched_2D_selected,
                                k1_parameter_baseline_2D_selected=k1_parameter_baseline_2D_selected,
                                A2_parameter_2D_selected=A2_parameter_2D_selected,
                                k2_parameter_enriched_2D_selected=k2_parameter_enriched_2D_selected,
                                k2_parameter_baseline_2D_selected=k2_parameter_baseline_2D_selected,
                                C_parameter_2D_selected=C_parameter_2D_selected)
            
        if 'psi0' not in encoding_df.columns:
            if set_offset_as_default:
                encoding_df['psi0'] = 0
            elif set_offset_as_total_reads:
                if offset_normalized:
                    encoding_df['psi0'] = np.log(enriched_read_counts + baseline_read_counts + offset_psuedocount)
                else:
                    encoding_df['psi0'] = np.log(encoding_df["total_reads"] + offset_psuedocount)
            elif set_offset_as_enriched:
                if offset_normalized:
                    encoding_df['psi0'] = np.log(enriched_read_counts + offset_psuedocount)
                else:
                    encoding_df['psi0'] = np.log(encoding_df[enriched_pop_df_reads_colname + "_raw"] + offset_psuedocount)
            elif set_offset_as_baseline:
                if offset_normalized:
                    encoding_df['psi0'] = np.log(baseline_read_counts + offset_psuedocount)
                else:
                    encoding_df['psi0'] = np.log(encoding_df[baseline_pop_df_reads_colname + "_raw"] + offset_psuedocount)
            elif set_offset_as_presort:
                if offset_normalized:
                    encoding_df['psi0'] = np.log(encoding_df[presort_pop_df_reads_colname] + offset_psuedocount)
                else:
                    encoding_df['psi0'] = np.log(encoding_df[presort_pop_df_reads_colname + "_raw"] + offset_psuedocount)
            assert np.all(np.isfinite(encoding_df['psi0'])), "NaN or inf psi0 offset values, perhaps due to 0 read counts. Consider setting cutoff_specifications or offset_psuedocount deepending on offset strategy (i.e. if set_offset_as_presort==True, then could set offset_psuedocount>0 or per_replicate_presort_condition_num_cutoff>0)"
        return encoding_df
    
    def __get_intercept_df(self, encoding_df_list: List[pd.DataFrame], experiment_id: Optional[int] = None) -> pd.DataFrame:
        assert len(encoding_df_list)>0, "Developer error: dataframe list should be greater than 0 as it was validated upstream"
        num_strata = len(encoding_df_list)
        
        intercept_postfix = None
        if experiment_id != None:
            # Experiment_id is provided, thus encoding_df_list is encodings for each rep, so put experiment_id in intercept preceeding replicate
            intercept_postfix = "_exp{}".format(experiment_id) + "_rep{}"
        else:
            # Experiment_id is NOT provided, thus encoding_df_list is encodings for each experiment, so just have experiment_id in intercept
            intercept_postfix = "_exp{}"
            
            
        # Iterate through each new intercept to add
        intercept_list = []
        for intercept_i in range(num_strata):
            intercept_i_list = []
            
            # Iterate through each strata
            for strata_i in range(num_strata):
                
                # Set intercept to be 1 for corresponding strata, else 0
                if intercept_i == strata_i:
                    intercept_i_list.extend(np.ones(encoding_df_list[strata_i].shape[0]))
                else:
                    intercept_i_list.extend(np.zeros(encoding_df_list[strata_i].shape[0]))
            intercept_list.append(intercept_i_list)

        intercept_df = pd.DataFrame(intercept_list).transpose().fillna(0)
        # POTENTIAL BUG 10/19/22: It is possible that "_" is not acceptable in intercept column.
        
        intercept_df.columns = ["intercept"  + intercept_postfix.format(strata_i) for strata_i in range(num_strata)]
        
        return intercept_df
    

@dataclass    
class MillipedeModelExperimentalGroup:
    
    experiments_inputdata: MillipedeInputDataExperimentalGroup
    device: MillipedeComputeDevice=MillipedeComputeDevice.CPU
    def __post_init__(self):
        self.millipede_model_specification_set_with_results = self.__run_all_models(experiments_inputdata=self.experiments_inputdata, device=self.device)
    
    def __run_all_models(self, experiments_inputdata: MillipedeInputDataExperimentalGroup, device: MillipedeComputeDevice=MillipedeComputeDevice.CPU) -> Mapping[str, MillipedeModelSpecificationResult]:
        millipede_model_specification_set_with_results: Mapping[str, MillipedeModelSpecificationResult] = {}
        millipede_model_specification_set_with_data = experiments_inputdata.millipede_model_specification_set_with_data
        print("Start model inference for all provided model specifications: {} total".format(len(millipede_model_specification_set_with_data)))
        millipede_model_specification_id: str # This is the ID for a model specification
        millipede_model_specification_tuple: Tuple[MillipedeModelSpecification, MillipedeInputData] # This is the model specification and corresponding model input data
        for index, (millipede_model_specification_id, millipede_model_specification_tuple) in enumerate(millipede_model_specification_set_with_data.items()):
            print("Starting model inference for model specification id {}/{}: {}".format(index+1, len(millipede_model_specification_set_with_data), millipede_model_specification_id))
            
            # Model specification
            millipede_model_specification: MillipedeModelSpecification = millipede_model_specification_tuple[0]
            
            # Model input data
            millipede_input_data: MillipedeInputData = millipede_model_specification_tuple[1]
            data: Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]] = millipede_input_data.data
            num_single_matrices = self.__count_single_matrices(data)
            
            print("Number of single matrices: {}".format(num_single_matrices))
            print("With {} model types, the total models to inference for this model specification: {}".format(len(millipede_model_specification.model_types), num_single_matrices*len(millipede_model_specification.model_types)))
            
            # Iterate through the data dependintwg on its list structure, run models, and add to set
            single_matrix_count = 0
            millipede_model_specification_result_input: Union[MillipedeModelSpecificationSingleMatrixResult, List[MillipedeModelSpecificationSingleMatrixResult], List[List[MillipedeModelSpecificationSingleMatrixResult]]] = None
            if isinstance(data, list):
                millipede_model_specification_result_input = []
                
                data: Union[List[pd.DataFrame], List[List[pd.DataFrame]]]
                for sub_data in data:
                    if isinstance(sub_data, list):
                        assert millipede_model_specification.replicate_merge_strategy == MillipedeReplicateMergeStrategy.SEPARATE and millipede_model_specification.experiment_merge_strategy == MillipedeExperimentMergeStrategy.SEPARATE, "Developer error: Millipede input data structure does not match with the merge strategies, data: List[List[pd.DataFrame]]; replicate_merge_strategy={}; experiment_merge_strategy={}".format(millipede_model_specification.replicate_merge_strategy, millipede_model_specification.experiment_merge_strategy)
                        
                        millipede_model_specification_single_matrix_result_sublist: List[MillipedeModelSpecificationSingleMatrixResult] = []
                        
                        millipede_model_specification_result_input: List[List[MillipedeModelSpecificationSingleMatrixResult]]
                        data: List[List[pd.DataFrame]]
                        sub_data: List[pd.DataFrame]
                        sub_sub_data: pd.DataFrame
                        for sub_sub_data in sub_data:
                            print("Running model(s) for single matrix index: {}/{}".format(single_matrix_count+1, num_single_matrices))
                            millipede_model_specification_single_matrix_result: MillipedeModelSpecificationSingleMatrixResult=self.__inference_model(full_data_design_matrix=sub_sub_data, 
                                       millipede_model_specification=millipede_model_specification,
                                       millipede_input_data=millipede_input_data,
                                       device=device)
                                
                            millipede_model_specification_single_matrix_result_sublist.append(millipede_model_specification_single_matrix_result)
                            single_matrix_count = single_matrix_count + 1
                        
                        millipede_model_specification_result_input.append(millipede_model_specification_single_matrix_result_sublist)
                    else:
                        millipede_model_specification_result_input: List[MillipedeModelSpecificationSingleMatrixResult]
                        data: List[pd.DataFrame]
                        sub_data: pd.DataFrame
                        print("Running model(s) for single matrix index: {}/{}".format(single_matrix_count+1, num_single_matrices))
                        millipede_model_specification_single_matrix_result: MillipedeModelSpecificationSingleMatrixResult=self.__inference_model(full_data_design_matrix=sub_data, 
                                       millipede_model_specification=millipede_model_specification,
                                       millipede_input_data=millipede_input_data,
                                       device=device)
                            
                        millipede_model_specification_result_input.append(millipede_model_specification_single_matrix_result)
                        single_matrix_count = single_matrix_count + 1
            else:
                millipede_model_specification_result_input: MillipedeModelSpecificationSingleMatrixResult
                data: pd.DataFrame
                print("Running model(s) for single matrix index: {}/{}".format(single_matrix_count+1, num_single_matrices))
                millipede_model_specification_single_matrix_result: MillipedeModelSpecificationSingleMatrixResult=self.__inference_model(full_data_design_matrix=data, 
                                       millipede_model_specification=millipede_model_specification,
                                       millipede_input_data=millipede_input_data,
                                       device=device)
                    
                millipede_model_specification_result_input = millipede_model_specification_single_matrix_result
                single_matrix_count = single_matrix_count + 1
        
            millipede_model_specification_result_object = MillipedeModelSpecificationResult(millipede_model_specification_result_input=millipede_model_specification_result_input,
                                                                                            millipede_model_specification=millipede_model_specification,
                                                                                            millipede_input_data=millipede_input_data)
            
            millipede_model_specification_set_with_results[millipede_model_specification_id] = millipede_model_specification_result_object
        
        return millipede_model_specification_set_with_results
            
    
    # Returns dictionary of models for each model_type for a single input design matrix
    def __inference_model(self, 
                          full_data_design_matrix: pd.DataFrame, 
                          millipede_model_specification: MillipedeModelSpecification,
                          millipede_input_data: MillipedeInputData, 
                          device: MillipedeComputeDevice) -> MillipedeModelSpecificationSingleMatrixResult:
        nucleotide_ids = [col for col in full_data_design_matrix.columns if ">" in col] # TODO 20221021 - There needs to be a better way at specifying what the features are
        intercept_columns = [col for col in full_data_design_matrix.columns if "intercept_" in col] 
        model_types: List[MillipedeModelType]= millipede_model_specification.model_types
        # Iterate through all model types and inference mdoel
        S = millipede_model_specification.S
        tau = millipede_model_specification.tau
        tau_intercept = millipede_model_specification.tau_intercept
        print("Iterating through all {} provided models: ".format(len(model_types), model_types))
        models: Mapping[MillipedeModelType, Union[NormalLikelihoodVariableSelector, BinomialLikelihoodVariableSelector, NegativeBinomialLikelihoodVariableSelector]] = {}
        for i, model_type in enumerate(millipede_model_specification.model_types):
            if model_type == MillipedeModelType.NORMAL:
                print("Preparing data for model {}, {}/{}".format(model_type.value, i+1, len(model_types)))
                required_columns = intercept_columns + nucleotide_ids + ['score'] 
                sub_data_design_matrix = full_data_design_matrix[required_columns]    
                
                normal_selector = NormalLikelihoodVariableSelector(sub_data_design_matrix, 
                                                                   response_column='score',
                                                                   assumed_columns=intercept_columns,
                                                                   prior='isotropic',
                                                                   S=S, 
                                                                   tau=tau,
                                                                   tau_intercept=tau_intercept,
                                                                   precision="double", 
                                                                   device=device.value)

                print("Running model {}".format(model_type.value))
                normal_selector.run(T=5000, T_burnin=500, verbosity='bar', seed=0)
                models[model_type] = normal_selector

            elif model_type == MillipedeModelType.NORMAL_SIGMA_SCALED:
                print("Preparing data for model {}, {}/{}".format(model_type.value, i+1, len(model_types)))
                required_columns = intercept_columns + nucleotide_ids + ['score', 'scale_factor']
                sub_data_design_matrix = full_data_design_matrix[required_columns]    
                
                normal_sigma_scaled_selector = NormalLikelihoodVariableSelector(sub_data_design_matrix,
                                                                                response_column='score',
                                                                                sigma_scale_factor_column='scale_factor',
                                                                                assumed_columns=intercept_columns,
                                                                                prior='isotropic',
                                                                                S=S, 
                                                                                tau=tau,
                                                                                tau_intercept=tau_intercept,
                                                                                precision="double", 
                                                                                device=device.value)

                print("Running model {}".format(model_type.value))
                normal_sigma_scaled_selector.run(T=5000, T_burnin=500, verbosity='bar', seed=0)
                models[model_type] = normal_sigma_scaled_selector

            elif model_type == MillipedeModelType.BINOMIAL:
                print("Preparing data for model {}, {}/{}".format(model_type.value, i+1, len(model_types)))
                required_columns = intercept_columns + nucleotide_ids + ['total_reads', millipede_input_data.enriched_pop_df_reads_colname]
                sub_data_design_matrix = full_data_design_matrix[required_columns]    
                binomial_selector = BinomialLikelihoodVariableSelector(sub_data_design_matrix, 
                                                                       response_column=millipede_input_data.enriched_pop_df_reads_colname,
                                                                       total_count_column='total_reads',
                                                                       assumed_columns=intercept_columns,
                                                                       S=S, 
                                                                       tau=tau,
                                                                       tau_intercept=tau_intercept,
                                                                       precision="double", 
                                                                       device=device.value)

                print("Running model {}".format(model_type.value))
                binomial_selector.run(T=5000, T_burnin=500, verbosity='bar', seed=0)
                models[model_type] = binomial_selector

            elif model_type == MillipedeModelType.NEGATIVE_BINOMIAL:
                print("Preparing data for model {}, {}/{}".format(model_type.value, i+1, len(model_types)))
                required_columns = intercept_columns + nucleotide_ids + [millipede_input_data.enriched_pop_df_reads_colname + "_raw", "psi0"]
                sub_data_design_matrix = full_data_design_matrix[required_columns]    
                negative_binomial_selector = NegativeBinomialLikelihoodVariableSelector(sub_data_design_matrix, 
                                                                       response_column=millipede_input_data.enriched_pop_df_reads_colname + "_raw",
                                                                       psi0_column='psi0',
                                                                       assumed_columns=intercept_columns,
                                                                       S=S, 
                                                                       tau=tau,
                                                                       tau_intercept=tau_intercept,
                                                                       precision="double", 
                                                                       device=device.value)

                print("Running model {}".format(model_type.value))
                negative_binomial_selector.run(T=5000, T_burnin=500, verbosity='bar', seed=0)
                models[model_type] = negative_binomial_selector
            else:
                logging.warning("Unsupported MillipedeModelType '{}', perhaps use a different supported model type".format(model_type))
        
        millipede_model_specification_single_matrix_result = MillipedeModelSpecificationSingleMatrixResult(millipede_model_specification_single_matrix_result=models)
        return millipede_model_specification_single_matrix_result
    
    def __count_single_matrices(self, data: Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]]) -> int:
        count = 0
        if isinstance(data, list):
            data: Union[List[pd.DataFrame], List[List[pd.DataFrame]]]
            for sub_data in data:
                if isinstance(sub_data, list):
                    data: List[List[pd.DataFrame]]
                    sub_data: List[pd.DataFrame]
                    sub_sub_data: pd.DataFrame
                    for sub_sub_data in sub_data:
                        count = count + 1
                else:
                    data: List[pd.DataFrame]
                    sub_data: pd.DataFrame
                    count = count + 1
        else:
            count = count + 1
        return count
    

    # ---------- top-level runner to process named dataset objects ----------
    def plot_millipede_score_correlation(self, name="dataset", joint_specification_id="joint_replicate_joint_experiment_models", per_experiment_specification_id="joint_replicate_per_experiment_models", save_pdf_prefix=None, reads_threshold=100):
        # ---------- utility functions (style similar to your example) ----------
        def get_nt_columns(df):
            """Return columns that look like nucleotide-change columns (contain '>')."""
            return [c for c in df.columns if '>' in c]

        def find_intercept_cols(df):
            """
            Find columns with names like 'intercept_expX_repY' and return dict:
            { exp_index(int) : [colname,...] } sorted by rep index.
            """
            intercept_cols = [c for c in df.columns if c.startswith('intercept_')]
            pattern = re.compile(r'intercept_exp(\d+)_rep(\d+)')
            out = {}
            for c in intercept_cols:
                m = pattern.search(c)
                if m:
                    exp_i = int(m.group(1))
                    rep_i = int(m.group(2))
                    out.setdefault(exp_i, []).append((rep_i, c))
            # sort by rep index and return names only
            for k in list(out.keys()):
                out[k] = [c for _, c in sorted(out[k])]
            return out

        def compute_edit_fraction_per_rep(df, nt_cols=None):
            """
            For every intercept rep column compute fraction of reads (weighted by #Reads_Presort_raw)
            that contain ABE edits (A>G or T>C), CBE edits (C>T or G>A), or any edit (any of nt_cols).
            Returns a DataFrame with index = rep_col and columns = ['ABE_frac','CBE_frac','any_frac','total_reads'].
            """
            if nt_cols is None:
                nt_cols = get_nt_columns(df)
            # patterns for ABE/CBE
            abe_mask_cols = [c for c in df.columns if ('A>G' in c) or ('T>C' in c)]
            cbe_mask_cols = [c for c in df.columns if ('C>T' in c) or ('G>A' in c)]
            intercept_map = find_intercept_cols(df)
            rows = []
            for exp_i, rep_cols in intercept_map.items():
                for rep_col in rep_cols:
                    mask = df[rep_col] == 1
                    total_reads = df.loc[mask, "#Reads_Presort_raw"].sum()
                    if total_reads == 0:
                        abe_frac = cbe_frac = any_frac = np.nan
                    else:
                        abe_reads = df.loc[mask, "#Reads_Presort_raw"][
                            (df.loc[mask, abe_mask_cols].sum(axis=1) > 0)
                        ].sum()
                        cbe_reads = df.loc[mask, "#Reads_Presort_raw"][
                            (df.loc[mask, cbe_mask_cols].sum(axis=1) > 0)
                        ].sum()
                        any_reads = df.loc[mask, "#Reads_Presort_raw"][
                            (df.loc[mask, nt_cols].sum(axis=1) > 0)
                        ].sum()
                        abe_frac = abe_reads / total_reads * 100.0
                        cbe_frac = cbe_reads / total_reads * 100.0
                        any_frac = any_reads / total_reads * 100.0
                    rows.append({
                        "rep_col": rep_col,
                        "exp": exp_i,
                        "ABE_pct": abe_frac,
                        "CBE_pct": cbe_frac,
                        "any_pct": any_frac,
                        "total_reads": total_reads
                    })
            return pd.DataFrame(rows).set_index("rep_col")

        # ---------- plotting helpers (pairwise with spearman + SD) ----------
        def corr_sd_text(x, y, **kws):
            ax = plt.gca()
            x = np.asarray(x); y = np.asarray(y)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() == 0:
                r, p, sd = np.nan, np.nan, np.nan
            else:
                r, p = spearmanr(x[m], y[m])
                sd = np.std(x[m] - y[m], ddof=1)
            ax.text(0.5, 0.65, f"r = {r:.2f}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, fontweight="bold")
            ax.text(0.5, 0.48, f"p = {p:.1g}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
            ax.text(0.5, 0.30, f"SD = {sd:.3g}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)

        def lower_scatter_simple(x, y, **kws):
            ax = plt.gca()
            ax.scatter(x, y, s=25, alpha=0.4, edgecolor=None)
            # dashed identity and zero lines
            lo = min(np.nanmin(x), np.nanmin(y))
            hi = max(np.nanmax(x), np.nanmax(y))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="k", alpha=0.7, linewidth=1)
            ax.axvline(0, linestyle="--", color="gray", alpha=0.6, linewidth=1)
            ax.axhline(0, linestyle="--", color="gray", alpha=0.6, linewidth=1)

        def pairwise_plot_scores(scores_df, title=None, save_pdf_handle=None, figsize=(0.6,0.6)):
            """
            scores_df: pandas DataFrame of shape (n_points, n_replicates) with numeric scores.
            Produces PairGrid with lower=scatter, diag=hist, upper=corr+sd.
            If save_pdf_handle provided, save figure there.
            """
            n_cols = scores_df.shape[1]
            if n_cols < 2:
                raise ValueError("Need at least 2 replicates for pairwise plotting")
            total_w, total_h = figsize
            height = total_h * n_cols
            aspect = (total_w * n_cols) / height
            g = sns.PairGrid(scores_df, diag_sharey=False, height=height, aspect=aspect)
            g.map_lower(lower_scatter_simple)
            g.map_diag(sns.histplot, bins=20, kde=False)
            g.map_upper(corr_sd_text)
            g.fig.suptitle(title or "", fontsize=14, fontweight="bold")
            g.fig.tight_layout(rect=[0, 0, 1, 0.95])
            if save_pdf_handle:
                save_pdf_handle.savefig(g.fig, bbox_inches="tight")
            else:
                plt.show()
            plt.close(g.fig)
            return g

        # ---------- core logic for building replicate score lists ----------
        def build_replicate_scores(df, nt_cols=None, reads_threshold=100):
            """
            For each intercept_expX_repY group, build a list of 'score' values across alleles:
            - group alleles by nt_cols (uses groupby(nt_cols))
            - require np.all(allele_group["#Reads_HbFHigh_raw"] + allele_group["#Reads_HbFLow_raw"] > reads_threshold)
            - from each allele group, take the 'score' value where intercept_col == 1 (one value per replicate)
            Returns dict: { exp_index: DataFrame(scores) } where each DataFrame has columns Replicate 1..N
            """
            if nt_cols is None:
                nt_cols = get_nt_columns(df)
            intercept_map = find_intercept_cols(df)
            out = {}
            # prepare grouped alleles
            grouped = df.groupby(nt_cols)
            for exp_i, rep_cols in intercept_map.items():
                rep_scores = {}  # rep_col -> list
                for rep_col in rep_cols:
                    rep_scores[rep_col] = []
                # iterate allele groups
                for allele, allele_group in grouped:
                    try:
                        # condition exactly as your original code: for all rows in allele_group the HbF reads sum > threshold
                        cond = np.all((allele_group["#Reads_HbFHigh_raw"] + allele_group["#Reads_HbFLow_raw"]) > reads_threshold)
                        if not cond:
                            continue
                    except Exception:
                        # if no HbF columns present or missing data, skip allele
                        continue
                    # for each replicate, pick the allele_group row with intercept==1 and grab its score (if exists)
                    for rep_col in rep_cols:
                        try:
                            val = allele_group.loc[allele_group[rep_col] == 1, "score"].iloc[0]
                            rep_scores[rep_col].append(val)
                        except Exception:
                            # missing value -> skip
                            pass
                # convert to DataFrame with columns ordered by rep index
                # rename columns to "Rep0", "Rep1", ...
                if rep_scores:
                    # determine rep order via numbers in names
                    rep_order = []
                    for rc in rep_cols:
                        m = re.search(r'rep(\d+)', rc)
                        rep_order.append((int(m.group(1)) if m else 0, rc))
                    rep_order = [c for _, c in sorted(rep_order)]

                    # build DataFrame from lists (lists may have different lengths -> align by index)
                    col_map = {}
                    max_len = 0

                    # Build mapping and track max length
                    for c in rep_order:
                        m = re.search(r'rep(\d+)', c)
                        rep_num = int(m.group(1)) if m else 0
                        vals = rep_scores[c]
                        col_map[f"Rep{rep_num}"] = vals
                        max_len = max(max_len, len(vals))

                    # Pad shorter lists with NaN so all columns are equal length
                    for k, v in col_map.items():
                        if len(v) < max_len:
                            col_map[k] = v + [np.nan] * (max_len - len(v))

                    # Now safely build DataFrame
                    scores_df = pd.DataFrame(col_map)
                    out[exp_i] = scores_df


            return out


        # try to fetch relevant dataframes
        try:
            joint_df = self.millipede_model_specification_set_with_results[joint_specification_id].millipede_input_data.data
        except Exception:
            joint_df = None
        try:
            per_exp_list = self.millipede_model_specification_set_with_results[per_experiment_specification_id].millipede_input_data.data
            # per_exp_list is expected to be a list-like (exp0, exp1, ..)
        except Exception:
            per_exp_list = None

        pdf_handle = None
        if save_pdf_prefix:
            pdf_handle = PdfPages(f"{save_pdf_prefix}_{name}.pdf")

        # function to do one dataframe full analysis (compute edit fractions and pairwise plots)
        def analyze_df(df, label):
            print(f"\n--- ANALYZING {name} :: {label} ---")
            nt_cols = get_nt_columns(df)
            print(f"Found {len(nt_cols)} nt columns (sample): {nt_cols[:6]}")
            # compute edit fractions per replicate
            edit_df = compute_edit_fraction_per_rep(df, nt_cols=nt_cols)
            print("Edit fractions per replicate (ABE / CBE / any %):")
            print(edit_df[["exp", "ABE_pct", "CBE_pct", "any_pct", "total_reads"]])
            # build score lists
            replicate_scores_dict = build_replicate_scores(df, nt_cols=nt_cols, reads_threshold=reads_threshold)
            # for each experiment create pairwise plot
            for exp_i, scores_df in replicate_scores_dict.items():
                if scores_df.shape[1] < 2:
                    print(f"Exp {exp_i}: less than 2 replicates with scores, skipping pairwise plot.")
                    continue
                title = f"{name} :: {label} :: experiment {exp_i} (reads_threshold={reads_threshold})"
                pairwise_plot_scores(scores_df, title=title, save_pdf_handle=pdf_handle)
                # also show small printout of number of alleles used per replicate
                counts = scores_df.notnull().sum()
                print(f"Exp {exp_i} replicate counts (alleles used):")
                print(counts)
            return edit_df, replicate_scores_dict

        results = {}
        # analyze joint_df if present
        if joint_df is not None:
            edit_df, rs = analyze_df(joint_df, joint_specification_id)
            results["joint"] = (edit_df, rs)
        # analyze per-experiment dfs if present
        if per_exp_list is not None:
            # per_exp_list might be list-like: iterate with index
            for idx, per_df in enumerate(per_exp_list):
                edit_df, rs = analyze_df(per_df, f"{per_experiment_specification_id}.exp{idx}")
                results[f"per_exp_{idx}"] = (edit_df, rs)

        if pdf_handle:
            pdf_handle.close()
            print(f"Saved pairwise plots to {save_pdf_prefix}_{name}.pdf")
        return results




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from typing import Union, List
from os.path import exists

parse_position = lambda feature: int(feature[:feature.index(">")-1])
parse_ref = lambda feature: feature[feature.index(">")-1:feature.index(">")]
parse_alt = lambda feature: feature[feature.index(">")+1:]


# TODO: Perhaps in the final library, this object can be created based on filenames (user-facing) or based on established pandas objects (internally)
@dataclass
class RawEncodingDataframesExperimentalGroup:
    
    
    # TODO set_variables_constructor and read_in_files_constructor will be classmethods as alternative contructors (factor methods) see https://www.programiz.com/python-programming/methods/built-in/classmethod
    # TODO: Reimpliment set_variables_constructor based on new input arguments from read_in_files_constructor
    def set_variables_constructor(self, 
           enriched_pop_encodings_df_experiment_list: List[List[pd.DataFrame]],
            baseline_pop_encodings_df_experiment_list: List[List[pd.DataFrame]], 
            experiment_labels: List[str],
            presort_pop_encodings_df_experiment_list: Optional[List[pd.DataFrame]] = None, 
            ctrl_pop_encodings_df_experiment_list: Optional[List[pd.DataFrame]] = None):
        self.enriched_pop_encodings_df_experiment_list = enriched_pop_encodings_df_experiment_list
        self.baseline_pop_encodings_df_experiment_list = baseline_pop_encodings_df_experiment_list
        self.presort_pop_encodings_df_experiment_list = presort_pop_encodings_df_experiment_list
        self.ctrl_pop_encodings_df_experiment_list = ctrl_pop_encodings_df_experiment_list
        
        self.__post_validate()
        
        return self
        
    def read_in_files_constructor(self, 
                                  enriched_pop_fn_encodings_experiment_list: List[str], 
                                  baseline_pop_fn_encodings_experiment_list: List[str], 
                                  experiment_labels: List[str],
                                  presort_pop_fn_encodings_experiment_list: Optional[List[str]] = None,
                                  ctrl_pop_fn_encodings: Optional[Union[list, str]] = None,
                                  ctrl_pop_labels: Optional[Union[list, str]]=None,
                                  reps:Optional[List[int]]=None):
                                  
        self.enriched_pop_encodings_df_experiment_list = enriched_pop_fn_encodings_experiment_list
        self.baseline_pop_fn_encodings_experiment_list = baseline_pop_fn_encodings_experiment_list
        self.presort_pop_fn_encodings_experiment_list = presort_pop_fn_encodings_experiment_list
        self.ctrl_pop_fn_encodings = ctrl_pop_fn_encodings
        
        '''
            Input pre-validation
        '''
        assert len(enriched_pop_fn_encodings_experiment_list) == len(baseline_pop_fn_encodings_experiment_list), "enriched_pop_encodings_df_list and baseline_pop_encodings_df_list must be same length"
        if presort_pop_fn_encodings_experiment_list != None:
            assert len(enriched_pop_fn_encodings_experiment_list) == len(presort_pop_fn_encodings_experiment_list), "If presort_pop_fn_encodings_experiment_list is provided, it must be the same length as enriched_pop_encodings_df_list and baseline_pop_encodings_df_list"
        
        print(enriched_pop_fn_encodings_experiment_list)
        enriched_pop_reps_list = []
        for fn in enriched_pop_fn_encodings_experiment_list:
            assert "{}" in fn, "Filename must have '{}' to replace with replicate ID, provided filename: " + str(fn)
            if reps is None:
                enriched_pop_reps = self.__check_file_locations(fn)
                enriched_pop_reps_list.append(enriched_pop_reps)
            else:
                self.__check_file_locations(fn, reps)
        
        baseline_pop_reps_list = []
        for fn in baseline_pop_fn_encodings_experiment_list:
            assert "{}" in fn, "Filename must have '{}' to replace with replicate ID, provided filename: " + str(fn)
            if reps is None:
                baseline_pop_reps = self.__check_file_locations(fn)
                baseline_pop_reps_list.append(baseline_pop_reps)
            else:
                self.__check_file_locations(fn, reps)
        
        if presort_pop_fn_encodings_experiment_list != None: 
            presort_pop_reps_list = []
            for fn in presort_pop_fn_encodings_experiment_list:
                assert "{}" in fn, "Filename must have '{}' to replace with replicate ID, provided filename: " + str(fn)  
                if reps is None:
                    presort_pop_reps = self.__check_file_locations(fn)
                    presort_pop_reps_list.append(presort_pop_reps)
                else:
                    self.__check_file_locations(fn, reps)

        if reps is None:
            assert enriched_pop_reps_list == baseline_pop_reps_list, f"Enriched and baseline filenames have different number of replicates. Enriched={enriched_pop_reps_list}, baseline={baseline_pop_reps_list}"
            reps=enriched_pop_reps_list
            print(f"Final inferred replicate list: {reps}")
            if presort_pop_fn_encodings_experiment_list != None: 
                assert presort_pop_reps_list == reps, f"Presort has different filename replicates compared to baseline and enriched samples. Presort={presort_pop_reps_list}, enriched/baseline={reps}"

        '''
            Since the control pop fn encoding has a flexible structure, must appropriately validate with recursive code below
        '''
        # Recursive function to check filename of all filenames in sup_ctrl_pop_fn_encodings
        def check_ctrl_pop_fn_encodings_file_locations(sup_ctrl_pop_fn_encodings: Union[list, str]):
            if isinstance(sup_ctrl_pop_fn_encodings, list):
                for subb_ctrl_pop_fn_encodings in sup_ctrl_pop_fn_encodings:
                    check_ctrl_pop_fn_encodings_file_locations(subb_ctrl_pop_fn_encodings)
            elif isinstance(sup_ctrl_pop_fn_encodings, str):
                self.__check_file_locations(sup_ctrl_pop_fn_encodings)
            else:
                # This should be caught by the @typechecked decorator, so this else block should never run 
                raise Exception("Filenames in ctrl_pop_fn_encodings must be of type string")
        # Recursive function to ensure shape of ctrl_pop_labels equals ctrl_pop_fn_encodings
        def check_ctrl_pop_labels_shape(sup_ctrl_pop_labels: Union[list, str], sup_ctrl_pop_fn_encodings: Union[list, str], depth:int=0, breadth:int=0):
            if isinstance(sup_ctrl_pop_labels, list):
                assert isinstance(sup_ctrl_pop_fn_encodings, list), "Entry in ctrl_pop_labels is list while the matching entry in ctrl_pop_fn_encodings is a string, at depth={}, breadth={}".format(depth, breadth)
                assert len(sup_ctrl_pop_labels) == len(sup_ctrl_pop_fn_encodings), "List entry in ctrl_pop_labels is not same length as list entry in ctrl_pop_fn_encodings, at depth={}, breadth={}".format(depth, breadth)
                for i, subb_ctrl_pop_labels in enumerate(sup_ctrl_pop_labels):
                    subb_ctrl_pop_fn_encodings = sup_ctrl_pop_fn_encodings[i]
                    check_ctrl_pop_labels_shape(subb_ctrl_pop_labels, subb_ctrl_pop_fn_encodings, depth=depth+1, breadth=i)
            elif isinstance(sup_ctrl_pop_labels, str):
                assert isinstance(sup_ctrl_pop_fn_encodings, str), "Entry in ctrl_pop_labels is string while the matching entry in ctrl_pop_fn_encodings is a list, at depth={}, breadth={}".format(depth, breadth)
                return # Passed, return back
            else:
                # This should be caught by the @typechecked decorator, so this else block should never run 
                raise Exception("Filenames in ctrl_pop_fn_encodings must be of type string")
        if ctrl_pop_fn_encodings != None:
            # Recursively check the filepaths in the provided controls
            check_ctrl_pop_fn_encodings_file_locations(ctrl_pop_fn_encodings)
            
            # Also check that the labels are provided and in the same shape as the ctrls:
            assert ctrl_pop_labels != None, "If ctrl_pop_fn_encodings filenames are provded, ctrl_pop_labels must be provided (make sure both are the same shape)"
            check_ctrl_pop_labels_shape(ctrl_pop_labels, ctrl_pop_fn_encodings)
            
                    
        '''
            Read in the files
        '''
        # Recursive function to read encodings
        def read_encodings_in_nested_list(sup_encoding_fn: Union[list, str], reps: Union[List[List[int]], List[int]]=None, _depth:int=0, _breadth:int=0):
            if isinstance(sup_encoding_fn, list):
                sup_encoding_df_list = []
                for i, subb_encoding_fn in enumerate(sup_encoding_fn):
                    sup_encoding_df_list.append(read_encodings_in_nested_list(subb_encoding_fn, reps[i], _depth=_depth+1, _breadth=i))
                return sup_encoding_df_list
            elif isinstance(sup_encoding_fn, str):
                sup_encoding_df_reps_list = []
                for rep in reps:
                    try:
                        sup_encoding_df_reps_list.append(pd.read_pickle(sup_encoding_fn.format(rep)))
                    except Exception as e:
                        raise Exception("Error reading encoding {} in provided position of filename list (depth={}, breadth={}); original exception: {}".format(subb_encoding_fn.format(rep, depth, breadth, str(e))))
                return sup_encoding_df_reps_list

        print("Reading enriched population...")
        self.enriched_pop_encodings_df_experiment_list: List[List[pd.Dataframe]] = read_encodings_in_nested_list(enriched_pop_fn_encodings_experiment_list, reps)
        print("Reading baseline population...")
        self.baseline_pop_encodings_df_experiment_list: List[List[pd.Dataframe]] = read_encodings_in_nested_list(baseline_pop_fn_encodings_experiment_list, reps)
        print("Reading presort population if provided...")
        if presort_pop_fn_encodings_experiment_list != None:
            self.presort_pop_encodings_df_experiment_list: List[List[pd.Dataframe]] = read_encodings_in_nested_list(presort_pop_fn_encodings_experiment_list, reps) 
        print("Reading control population if provided...")
        if ctrl_pop_fn_encodings != None:
            self.ctrl_pop_encodings_df_list: list = read_encodings_in_nested_list(ctrl_pop_fn_encodings) 
        
        self.__post_validate()
        return self
        
        
    def __post_validate(self):
        # TODO: This is just checking that the experiment size is the same, but not the replicates. It should be fine, in fact this validation may not be needed because it is highly unlikely that an assertion will be thrown
        '''
            Output post-validation
        '''
        assert len(self.enriched_pop_encodings_df_experiment_list) == len(self.baseline_pop_encodings_df_experiment_list), "List of final encoding_dfs for enriched and baseline are not the same, despite input sizes being the same. Perhaps an issue in string formatting."
        if hasattr(self, "presort_pop_encodings_df_experiment_list"):
            assert len(self.enriched_pop_encodings_df_experiment_list) == len(self.presort_pop_encodings_df_experiment_list), "List of final encoding_dfs for enriched/baseline and presort are not the same, despite input sizes being the same. Perhaps an issue in string formatting."
        print("Passed post-validation")
        
        self.validated = True
        
    def __check_file_locations(self, fn: str, reps: Optional[List[int]]=None) -> Optional[List[int]]:
        if reps != None:
            for rep in reps:
                assert exists(fn.format(rep)), "File not found: " + fn.format(rep)
        else:
            file_available = True
            reps = []
            rep = 0
            while file_available:
                if exists(fn.format(rep)):
                    reps.append(rep)
                    rep = rep + 1
                else:
                    if len(reps) == 0:
                        raise Exception("No files found. Make sure input filename contains {} to insert replicate number")
                    file_available = False
                    break
            return reps
            #exists(fn), "File not found: " + fn

    
    def compute_pairwise_correlations(self, read_threshold=0):
        """
        Computes Pearson correlation between all samples across populations/replicates.
        Includes WT controls.
        Filters out alleles with summed reads (across the pair) < read_threshold.
        Returns a correlation matrix (pd.DataFrame) and the dictionary of sample Series.
        """
        # ============================================================
        # STEP 1. Extract allele count Series from each sample
        # ============================================================

        def get_sample_series(df):
            """
            Extracts a Series of allele counts indexed by allele_id for one sample.
            Removes WT (all 0) alleles.
            """
            allele_cols = [c for c in df.columns if ">" in c]
            count_col = [c for c in df.columns if "#Reads" in c][0]

            # Drop WT alleles (all mutation indicators are 0)
            is_wt = (df[allele_cols] == 0).all(axis=1)
            df = df.loc[~is_wt].copy()

            # Create allele identifier string (or could use tuple)
            df["allele_id"] = df[allele_cols].astype(str).agg("".join, axis=1)

            # Group by allele_id in case duplicates exist, and sum counts
            s = df.groupby("allele_id")[count_col].sum()

            return s


        populations = [
            "baseline_pop_encodings_df_experiment_list",
            "enriched_pop_encodings_df_experiment_list",
            "presort_pop_encodings_df_experiment_list",
            "ctrl_pop_encodings_df_experiment_list",  # ✅ Include WT controls
        ]

        sample_series_dict = {}
        for pop in populations:
            pop_list = getattr(self, pop)
            for exp_i, exp_reps in enumerate(pop_list):
                for rep_i, df in enumerate(exp_reps):
                    label = f"{pop.split('_')[0]}_exp{exp_i}_rep{rep_i}"
                    s = get_sample_series(df)
                    sample_series_dict[label] = s

        sample_names = list(sample_series_dict.keys())
        print(f"Loaded {len(sample_names)} total samples (including WT).")
        print(f"📈 Computing {len(sample_names)*(len(sample_names)-1)//2} pairwise correlations...\n")

        results = []

        for (a, b) in itertools.combinations(sample_names, 2):
            s1 = sample_series_dict[a]
            s2 = sample_series_dict[b]

            # Merge only this pair
            merged = pd.merge(
                s1.rename("a"), s2.rename("b"),
                left_index=True, right_index=True, how="inner"
            )

            # Apply read threshold filter (sum of both samples)
            merged["total_reads"] = merged["a"] + merged["b"]
            merged = merged.loc[merged["total_reads"] >= read_threshold]

            n_alleles = len(merged)
            corr = np.nan
            if n_alleles > 1:
                corr = merged["a"].corr(merged["b"])

            print(f"   {a} ↔ {b}: r={corr:.3f} ({n_alleles} alleles compared after filtering < {read_threshold})")
            results.append((a, b, corr, n_alleles))

        # Convert results to symmetric DataFrame
        corr_df = pd.DataFrame(index=sample_names, columns=sample_names, dtype=float)
        for a, b, r, n in results:
            corr_df.loc[a, b] = r
            corr_df.loc[b, a] = r
        np.fill_diagonal(corr_df.values, 1.0)

        self.corr_df = corr_df
        self.sample_series_dict = sample_series_dict

    def plot_correlation_heatmap(self, figsize=(9, 9)):
        """
        Clustered heatmap with row/col colors for both population and replicate.
        """

        # --- CLEANING STEP (fix for ValueError) ---
        corr_df = self.corr_df.copy().astype(float)
        corr_df = corr_df.replace([np.inf, -np.inf], np.nan)
        corr_df = corr_df.fillna(0)  # or np.nanmean alternative

        # Define colors for populations
        pop_colors = {
            "baseline": "#1f77b4",
            "enriched": "#2ca02c",
            "presort": "#d62728",
            "ctrl": "#ffbb78",  # add WT control color
        }

        # Define colors for replicates (cycle through 6 for example)
        replicate_colors = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

        populations = []
        replicates = []

        for s in corr_df.index:
            # population
            pop = next((k for k in pop_colors if k in s), None)
            populations.append(pop_colors[pop] if pop else "#808080")

            # replicate number
            match = re.search(r"rep(\d+)", s)
            if match:
                rep_i = int(match.group(1))
                replicates.append(replicate_colors[rep_i % len(replicate_colors)])
            else:
                replicates.append("#808080")

        # Combine into a DataFrame for seaborn
        row_colors = pd.DataFrame({
            "Population": populations,
            "Replicate": replicates
        }, index=corr_df.index)

        print(f"\n📊 Plotting clustered heatmap with population + replicate colors (vmin=-1, vmax=1)...")

        g = sns.clustermap(
            corr_df,
            cmap="Reds",
            figsize=figsize,
            vmin=0.25,
            vmax=1,
            linewidths=0.3,
            row_colors=row_colors,
            col_colors=row_colors,
            cbar_kws={"label": "Pearson r"},
            method="average",
            metric="correlation",
        )

        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        plt.show()



from typing import Callable

@dataclass
class EncodingEditingFrequenciesExperimentalGroup:
    raw_encodings: RawEncodingDataframesExperimentalGroup
    """
        Class containing editing frequencies for encoding group
        
        TODO 20221017 DONE - creating "encoding group" object (of all population) in single container. 
        TODO 20221018 - pass in a specific variant type
    """
    def __post_init__(self):
        
        '''
            Input pre-validation
        '''
        #TODO: For some reason, can access __validated
        #assert raw_encodings.__validated == True, "Raw encoding object provided is not valid, ensure object properly created from its constructor"
        
        '''
            Calculate per-position editing frequency 
        '''
        self.enriched_pop_encoding_editing_freq_experiment_list: List[List[pd.Series]] = [[self.__generate_per_position_editing_frequency(encoding_df) for encoding_df in encoding_reps_df] for encoding_reps_df in self.raw_encodings.enriched_pop_encodings_df_experiment_list]
        self.baseline_pop_encoding_editing_freq_experiment_list: List[List[pd.Series]] = [[self.__generate_per_position_editing_frequency(encoding_df) for encoding_df in encoding_reps_df] for encoding_reps_df in self.raw_encodings.baseline_pop_encodings_df_experiment_list]
            
        self.enriched_pop_encoding_editing_per_variant_freq_experiment_list: List[List[pd.Series]] = [[self.__generate_per_variant_editing_frequency(encoding_df) for encoding_df in encoding_reps_df] for encoding_reps_df in self.raw_encodings.enriched_pop_encodings_df_experiment_list]
        self.baseline_pop_encoding_editing_per_variant_freq_experiment_list: List[List[pd.Series]] = [[self.__generate_per_variant_editing_frequency(encoding_df) for encoding_df in encoding_reps_df] for encoding_reps_df in self.raw_encodings.baseline_pop_encodings_df_experiment_list]
        
        if hasattr(self.raw_encodings, "presort_pop_encodings_df_experiment_list"):
            self.presort_pop_encoding_editing_freq_experiment_list: List[List[pd.Series]] = [[self.__generate_per_position_editing_frequency(encoding_df) for encoding_df in encoding_reps_df] for encoding_reps_df in self.raw_encodings.presort_pop_encodings_df_experiment_list] 
            self.presort_pop_encoding_editing_per_variant_freq_experiment_list: List[List[pd.Series]] = [[self.__generate_per_variant_editing_frequency(encoding_df) for encoding_df in encoding_reps_df] for encoding_reps_df in self.raw_encodings.presort_pop_encodings_df_experiment_list] 
        
        if hasattr(self.raw_encodings, "ctrl_pop_encodings_df_experiment_list"):
            self.ctrl_pop_encoding_editing_freq_experiment_list: List[List[pd.Series]] = [[self.__generate_per_position_editing_frequency(encoding_df) for encoding_df in encoding_reps_df] for encoding_reps_df in self.raw_encodings.ctrl_pop_encodings_df_experiment_list] 
            self.ctrl_pop_encoding_editing_per_variant_freq_experiment_list: List[List[pd.Series]] = [[self.__generate_per_variant_editing_frequency(encoding_df) for encoding_df in encoding_reps_df] for encoding_reps_df in self.raw_encodings.ctrl_pop_encodings_df_experiment_list] 
            
        
        '''
            Calculate average frequency across replicates
        '''
        def generate_editing_freq_avg_dict(editing_freq_experiment_list: List[List[pd.Series]]):
            editing_freq_avg_dict: Mapping[int, Union[List[pd.Series], pd.Series]] = {}
            editing_freq_avg_dict[1] = [sum(editing_freq_list) / len(editing_freq_list) for editing_freq_list in editing_freq_experiment_list]
            flattened_editing_freq_list = [editing_freq_series for editing_freq_list in editing_freq_experiment_list for editing_freq_series in editing_freq_list]
            editing_freq_avg_dict[0] = sum(flattened_editing_freq_list) / len(flattened_editing_freq_list)
            return editing_freq_avg_dict
        
        self.enriched_pop_encoding_editing_freq_avg = generate_editing_freq_avg_dict(self.enriched_pop_encoding_editing_freq_experiment_list)
        self.enriched_pop_encoding_editing_per_variant_freq_avg = generate_editing_freq_avg_dict(self.enriched_pop_encoding_editing_per_variant_freq_experiment_list)
        
        self.baseline_pop_encoding_editing_freq_avg = generate_editing_freq_avg_dict(self.baseline_pop_encoding_editing_freq_experiment_list)
        self.baseline_pop_encoding_editing_per_variant_freq_avg = generate_editing_freq_avg_dict(self.baseline_pop_encoding_editing_per_variant_freq_experiment_list)
        
        if hasattr(self, "presort_pop_encoding_editing_freq_experiment_list"):
            self.presort_pop_encoding_editing_freq_avg = generate_editing_freq_avg_dict(self.presort_pop_encoding_editing_freq_experiment_list)
            self.presort_pop_encoding_editing_per_variant_freq_avg = generate_editing_freq_avg_dict(self.presort_pop_encoding_editing_per_variant_freq_experiment_list)
            
        
        if hasattr(self, "ctrl_pop_encodings_df_experiment_list"):
            self.ctrl_pop_encoding_editing_freq_avg = generate_editing_freq_avg_dict(self.ctrl_pop_encoding_editing_freq_experiment_list)
            self.ctrl_pop_encoding_editing_per_variant_freq_avg = generate_editing_freq_avg_dict(self.ctrl_pop_encoding_editing_per_variant_freq_experiment_list)

        self.__validated = True
    
    def __generate_per_position_editing_frequency(self, encoding_df: pd.DataFrame) -> pd.Series:
        # Generate encoding-only DF and read count series
        nt_columns_indicator = pd.Series([">" in col for col in encoding_df.columns])
        position_series = pd.Series([parse_position(col) for col in encoding_df.columns[nt_columns_indicator]])
        encoding_only_df = encoding_df.loc[:, encoding_df.columns[nt_columns_indicator]]
        reads_df = encoding_df.loc[:, encoding_df.columns[~nt_columns_indicator]]
        assert reads_df.shape[1] == 1
        reads_series = reads_df.iloc[:, 0]

        # Generate per position counts by iterating through positions, subsetting the encoding by columns, and calculating frequency
        read_series_per_position = [reads_series[encoding_only_df.loc[:, encoding_only_df.columns[position_series == position]].sum(axis=1)>0] for position in position_series.unique()]
        encoding_df_position_collapsed_freq = pd.Series([sum(read_series_subset)/sum(reads_series) for read_series_subset in read_series_per_position], index=position_series.unique())
        return encoding_df_position_collapsed_freq
    
    def __generate_per_variant_editing_frequency(self, encoding_df: pd.DataFrame) -> pd.Series:
        # Generate encoding-only DF and read count series
        nt_columns_indicator = pd.Series([">" in col for col in encoding_df.columns])
        encoding_only_df = encoding_df.loc[:, encoding_df.columns[nt_columns_indicator]]
        reads_df = encoding_df.loc[:, encoding_df.columns[~nt_columns_indicator]]
        assert reads_df.shape[1] == 1
        reads_series = reads_df.iloc[:, 0]

        # Generate per variant frequency by multiplying 1/0 encoding by read series, then calculating frequency
        encoding_only_reads_mul_df = encoding_only_df.mul(reads_series, axis=0)
        encoding_df_freq = encoding_only_reads_mul_df.sum(axis=0) / sum(reads_series)
        return encoding_df_freq