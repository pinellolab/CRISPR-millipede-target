"""
Encoding data handling and editing frequency analysis.

This module manages raw allele-level encoding data, computes editing frequencies
at both position and variant levels, and provides visualization and analysis tools
for encoding properties across replicates and conditions.

Key Classes:
- RawEncodingDataframesExperimentalGroup: Loading and correlation analysis of raw encodings
- EncodingEditingFrequenciesExperimentalGroup: Calculation of per-position and per-variant frequencies

Helper Functions:
- parse_position: Extract position from feature string
- parse_ref: Extract reference base from feature string
- parse_alt: Extract alternate base from feature string
"""

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
from typing import Callable

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