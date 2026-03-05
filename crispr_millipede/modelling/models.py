import numpy as np
from scipy.optimize import curve_fit
import torch
from millipede import NormalLikelihoodVariableSelector
from millipede import BinomialLikelihoodVariableSelector
from millipede import NegativeBinomialLikelihoodVariableSelector
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools
import crispr_shrinkage

import logging

from os.path import exists

from dataclasses import dataclass
from typing import Union, List, Mapping, Tuple, Optional, Dict
from functools import partial
from typeguard import typechecked
from enum import Enum
from collections import defaultdict

import re
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages

from .models_inputs import *

from .pydeseq import run_pydeseq2

from .input_data import MillipedeInputDataExperimentalGroup

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
    def plot_millipede_score_correlation(self, presort_colname: str, enriched_colname: str, baseline_colname: str, name="dataset", joint_specification_id="joint_replicate_joint_experiment_models", per_experiment_specification_id="joint_replicate_per_experiment_models", save_pdf_prefix=None, reads_threshold=100):
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
                    total_reads = df.loc[mask, presort_colname].sum()
                    if total_reads == 0:
                        abe_frac = cbe_frac = any_frac = np.nan
                    else:
                        abe_reads = df.loc[mask, presort_colname][
                            (df.loc[mask, abe_mask_cols].sum(axis=1) > 0)
                        ].sum()
                        cbe_reads = df.loc[mask, presort_colname][
                            (df.loc[mask, cbe_mask_cols].sum(axis=1) > 0)
                        ].sum()
                        any_reads = df.loc[mask, presort_colname][
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
                        cond = np.all((allele_group[enriched_colname] + allele_group[baseline_colname]) > reads_threshold)
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
