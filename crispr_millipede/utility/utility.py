import numpy as np
import torch
from millipede import NormalLikelihoodVariableSelector
from millipede import BinomialLikelihoodVariableSelector
from millipede import NegativeBinomialLikelihoodVariableSelector
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import logging

from os.path import exists

from dataclasses import dataclass
from typing import Union, List, Mapping, Tuple, Optional
from functools import partial
from typeguard import typechecked
from enum import Enum
from collections import defaultdict

from ..modelling.models_inputs import *


@dataclass
class MillipedeInputDataExperimentalGroup:
    data_directory: str
    enriched_pop_fn_experiment_list: List[str]
    enriched_pop_df_reads_colname: str
    baseline_pop_fn_experiment_list: List[str]
    baseline_pop_df_reads_colname: str
    experiment_labels: List[str]
    reps: List[int]
    millipede_model_specification_set: Mapping[str, MillipedeModelSpecification]
    presort_pop_fn_experiment_list: Optional[List[str]] = None
    presort_pop_df_reads_colname: Optional[str] = None

    """
        Generates the MillipedeInputData objects provided MillipedeModelSpecifications and other relevant parameters such as filepaths to the data tables, read thresholds, and labels.
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
        
        
        __get_data_partial = partial(
            self.__get_data,
            data_directory=self.data_directory, 
            enriched_pop_fn_experiment_list=self.enriched_pop_fn_experiment_list, 
            enriched_pop_df_reads_colname=self.enriched_pop_df_reads_colname, 
            baseline_pop_fn_experiment_list=self.baseline_pop_fn_experiment_list, 
            baseline_pop_df_reads_colname=self.baseline_pop_df_reads_colname, 
            presort_pop_fn_experiment_list=self.presort_pop_fn_experiment_list, 
            presort_pop_df_reads_colname=self.presort_pop_df_reads_colname, 
            reps=self.reps
        )
        
    """ 
    Function to process the encoding dataframe (from encode pipeline script) and create design matrix for milliped
    """
    def __get_data(self, 
                   data_directory: str, 
                   enriched_pop_fn_experiment_list: List[str], 
                   enriched_pop_df_reads_colname: str, 
                   baseline_pop_fn_experiment_list: List[str], 
                   baseline_pop_df_reads_colname: str, 
                   presort_pop_fn_experiment_list: Optional[List[str]], 
                   presort_pop_df_reads_colname: Optional[str], 
                   reps: List[int], 
                   #replicate_merge_strategy:MillipedeReplicateMergeStrategy, 
                   #experiment_merge_strategy:MillipedeExperimentMergeStrategy,
                   #cutoff_specification: MillipedeCutoffSpecification,
                   #design_matrix_processing_specification: MillipedeDesignMatrixProcessingSpecification,
                   #shrinkage_input: Union[MillipedeShrinkageInput, None]
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
            merged_experiment_df_list: Union[List[pd.DataFrame], List[List[pd.DataFrame]]] = []
            # Iterate through the experiments
            for experiment_index in range(len(enriched_pop_fn_experiment_list)):
                # Get the enriched_population and baseline_population for the experiment
                enriched_pop_exp_fn = enriched_pop_fn_experiment_list[experiment_index]
                baseline_pop_exp_fn = baseline_pop_fn_experiment_list[experiment_index]
                
                presort_pop_exp_fn = None
                if presort_pop_fn_experiment_list is not None:
                    presort_pop_exp_fn = presort_pop_fn_experiment_list[experiment_index]
            
                # Iterate through each replicate of the experiment
                # type List[pd.DataFrame] if relicates are combined
                # type List[List[pd.DataFrame]] if replicates are separate
                exp_merged_rep_df_list: List[pd.DataFrame] = []
                for rep in reps:
                    '''
                        Check file directories
                    '''
                    enriched_pop_full_fn_exp_rep = (data_directory + '/' + enriched_pop_exp_fn).format(rep)
                    baseline_pop_full_fn_exp_rep = (data_directory + '/' + baseline_pop_exp_fn).format(rep)
                    
                    presort_pop_full_fn_exp_rep = None
                    if presort_pop_exp_fn is not None:
                        presort_pop_full_fn_exp_rep = (data_directory + '/' + presort_pop_exp_fn).format(rep)
                        assert exists(presort_pop_full_fn_exp_rep), "File not found: {}".format(presort_pop_full_fn_exp_rep)

                    assert exists(enriched_pop_full_fn_exp_rep), "File not found: {}".format(enriched_pop_full_fn_exp_rep)
                    assert exists(baseline_pop_full_fn_exp_rep), "File not found: {}".format(baseline_pop_full_fn_exp_rep)

                    '''
                        Read in dataframes
                    '''
                    enriched_pop_exp_rep_df = pd.read_csv(enriched_pop_full_fn_exp_rep, sep='\t').fillna(value=0.0)
                    enriched_pop_nt_columns = [col for col in enriched_pop_exp_rep_df.columns if ">" in col]
                    enriched_pop_exp_rep_df = enriched_pop_exp_rep_df[enriched_pop_nt_columns + [enriched_pop_df_reads_colname]]


                    baseline_pop_exp_rep_df = pd.read_csv(baseline_pop_full_fn_exp_rep, sep='\t').fillna(value=0.0)
                    baseline_pop_nt_columns = [col for col in baseline_pop_exp_rep_df.columns if ">" in col]
                    baseline_pop_exp_rep_df = baseline_pop_exp_rep_df[baseline_pop_nt_columns + [baseline_pop_df_reads_colname]]

                    assert set(enriched_pop_nt_columns) == set(baseline_pop_nt_columns), "Nucleotide columns between enriched and baseline dataframes must be equivalent - are these screening the same regions?"
                    
                    presort_pop_exp_rep_df = None
                    if presort_pop_full_fn_exp_rep is not None:
                        presort_pop_exp_rep_df = pd.read_csv(presort_pop_full_fn_exp_rep, sep='\t').fillna(value=0.0)
                        presort_pop_nt_columns = [col for col in presort_pop_exp_rep_df.columns if ">" in col]
                        presort_pop_exp_rep_df = presort_pop_exp_rep_df[presort_pop_nt_columns + [presort_pop_df_reads_colname]]
                        assert set(enriched_pop_nt_columns) == set(presort_pop_nt_columns), "Nucleotide columns between presort and the  enriched/baseline dataframes must be equivalent - are these screening the same regions?"


                    
                    nucleotide_ids = enriched_pop_nt_columns

                    # Concat the enriched and baseline population dataframes together
                    if presort_pop_full_fn_exp_rep is not None:
                        merged_exp_rep_df: pd.DataFrame = pd.concat([enriched_pop_exp_rep_df, baseline_pop_exp_rep_df, presort_pop_exp_rep_df]).groupby(nucleotide_ids, as_index=False).sum()
                    else:
                        merged_exp_rep_df: pd.DataFrame = pd.concat([enriched_pop_exp_rep_df, baseline_pop_exp_rep_df]).groupby(nucleotide_ids, as_index=False).sum()

                    # filter based on the per_replicate_each_condition_num_cutoff
                    merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df[baseline_pop_df_reads_colname] >= cutoff_specification.per_replicate_each_condition_num_cutoff]
                    merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df[enriched_pop_df_reads_colname] >= cutoff_specification.per_replicate_each_condition_num_cutoff]
                    if presort_pop_full_fn_exp_rep is not None:
                        merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df[presort_pop_df_reads_colname] >= cutoff_specification.per_replicate_presort_condition_num_cutoff]
                    merged_exp_rep_df['total_reads'] = merged_exp_rep_df[baseline_pop_df_reads_colname] + merged_exp_rep_df[enriched_pop_df_reads_colname]

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
                    merged_exp_reps_df = merged_exp_reps_df.groupby(nucleotide_ids, as_index=False).filter(partial(per_condition_reads_filter(baseline_pop_df_reads_colname, cutoff_specification.baseline_pop_per_condition_each_replicate_num_cutoff, cutoff_specification.baseline_pop_per_condition_acceptable_rep_count)))
                if (cutoff_specification.enriched_pop_per_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.enriched_pop_per_condition_acceptable_rep_count > 0):
                    print(f"Running enriched per-condition filtering with num_cutoff={cutoff_specification.enriched_pop_per_condition_each_replicate_num_cutoff} and acceptable_rep_count={cutoff_specification.enriched_pop_per_condition_acceptable_rep_count}")
                    merged_exp_reps_df = merged_exp_reps_df.groupby(nucleotide_ids, as_index=False).filter(partial(per_condition_reads_filter(enriched_pop_df_reads_colname, cutoff_specification.enriched_pop_per_condition_each_replicate_num_cutoff, cutoff_specification.enriched_pop_per_condition_acceptable_rep_count)))
                if (cutoff_specification.presort_pop_per_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.presort_pop_per_condition_acceptable_rep_count > 0):
                    print(f"Running enriched per-condition filtering with num_cutoff={cutoff_specification.presort_pop_per_condition_each_replicate_num_cutoff} and acceptable_rep_count={cutoff_specification.presort_pop_per_condition_acceptable_rep_count}")
                    merged_exp_reps_df = merged_exp_reps_df.groupby(nucleotide_ids, as_index=False).filter(partial(per_condition_reads_filter(presort_pop_df_reads_colname, cutoff_specification.presort_pop_per_condition_each_replicate_num_cutoff, cutoff_specification.presort_pop_per_condition_acceptable_rep_count)))


                # All-condition filtering
                if ((cutoff_specification.baseline_pop_all_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.baseline_pop_all_condition_acceptable_rep_count > 0)) | ((cutoff_specification.enriched_pop_all_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.enriched_pop_all_condition_acceptable_rep_count > 0)) | ((cutoff_specification.presort_pop_all_condition_each_replicate_num_cutoff > 0) and (cutoff_specification.presort_pop_all_condition_acceptable_rep_count > 0)) :
                    print(f"Running all-condition filtering with enriched_num_cutoff={cutoff_specification.enriched_pop_all_condition_each_replicate_num_cutoff}, enriched_acceptable_rep_count={cutoff_specification.enriched_pop_all_condition_acceptable_rep_count}, baseline_num_cutoff={cutoff_specification.baseline_pop_all_condition_each_replicate_num_cutoff}, baseline_acceptable_rep_count={cutoff_specification.baseline_pop_all_condition_acceptable_rep_count}, presort_num_cutoff={cutoff_specification.presort_pop_all_condition_each_replicate_num_cutoff}, presort_acceptable_rep_count={cutoff_specification.presort_pop_all_condition_acceptable_rep_count}")

                    def all_condition_filter_func(df: pd.DataFrame):
                        return per_condition_reads_filter(baseline_pop_df_reads_colname, cutoff_specification.baseline_pop_all_condition_each_replicate_num_cutoff, cutoff_specification.baseline_pop_all_condition_acceptable_rep_count, df) | per_condition_reads_filter(enriched_pop_df_reads_colname, cutoff_specification.enriched_pop_all_condition_each_replicate_num_cutoff, cutoff_specification.enriched_pop_all_condition_acceptable_rep_count, df) | per_condition_reads_filter(presort_pop_df_reads_colname, cutoff_specification.presort_pop_all_condition_each_replicate_num_cutoff, cutoff_specification.presort_pop_all_condition_acceptable_rep_count, df)
                    
                    merged_exp_reps_df = merged_exp_reps_df.groupby(nucleotide_ids, as_index=False).filter(all_condition_filter_func)

                # De-concatenate back into separate replicate by groupby on temporary rep_i column
                exp_merged_rep_df_list = [merged_exp_rep_df for _, merged_exp_rep_df in merged_exp_reps_df.groupby("rep_i")]

                '''
                    Perform normalization after filtering
                '''
                def normalize_func(merged_exp_rep_df):
                    merged_exp_rep_normalized_df: pd.DataFrame = self.__normalize_counts(merged_exp_rep_df, enriched_pop_df_reads_colname, baseline_pop_df_reads_colname, nucleotide_ids, design_matrix_processing_specification.wt_normalization, design_matrix_processing_specification.total_normalization, presort_pop_df_reads_colname) 
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
                        df = df.rename(columns={enriched_pop_df_reads_colname: enriched_pop_df_reads_colname+"_rep{}".format(rep_i), baseline_pop_df_reads_colname: baseline_pop_df_reads_colname+"_rep{}".format(rep_i)})
                        wt_allele_rep_df_renamed.append(df)
                    
                    # Group by allele
                    nucleotide_ids = [col for col in wt_allele_rep_df_renamed[0].columns if ">" in col]
                    wt_allele_rep_df_merged = pd.concat(wt_allele_rep_df).groupby(nucleotide_ids, as_index=False).sum() # This is for the final dataframe
                    wt_allele_rep_df_renamed_merged = pd.concat(wt_allele_rep_df_renamed).groupby(nucleotide_ids, as_index=False)
                    
                    
                    
                    negative_guides = []
                    for index, (name, group) in enumerate(wt_allele_rep_df_renamed_merged):
                        group_noNaN = group.fillna(0)
                        sample_population_raw_count_reps_observation = np.asarray([group_noNaN[enriched_pop_df_reads_colname+"_rep{}".format(rep_i)].sum() for rep_i in reps])
                        control_population_raw_count_reps_observation = np.asarray([group_noNaN[baseline_pop_df_reads_colname+"_rep{}".format(rep_i)].sum() for rep_i in reps])
                        # TODO: Later can add more info to guide, i.e. the allele. But setting the identifer as the df index is good and possibly sufficient.
                        guide = crispr_shrinkage.Guide(identifier="negative_{}".format(index), position=None, sample_population_raw_count_reps=sample_population_raw_count_reps_observation, control_population_raw_count_reps=control_population_raw_count_reps_observation, is_explanatory=True)
                        negative_guides.append(guide)
                        
                    
                    
                    # Get alleles that are mutated
                    mut_allele_rep_df = [merged_rep_df[merged_rep_df[nucleotide_ids].sum(axis=1) > 0] for merged_rep_df in exp_merged_rep_df_list]
                    
                    # Rename the dataframe to differentiate counts between reps
                    mut_allele_rep_df_renamed = []
                    for rep_i, df in enumerate(mut_allele_rep_df):
                        df = df.rename(columns={enriched_pop_df_reads_colname: enriched_pop_df_reads_colname+"_rep{}".format(rep_i), baseline_pop_df_reads_colname: baseline_pop_df_reads_colname+"_rep{}".format(rep_i)})
                        mut_allele_rep_df_renamed.append(df)
                    
                    # Group by allele
                    nucleotide_ids = [col for col in mut_allele_rep_df_renamed[0].columns if ">" in col]
                    mut_allele_rep_df_merged = pd.concat(mut_allele_rep_df).groupby(nucleotide_ids, as_index=False).sum()
                    mut_allele_rep_df_renamed_merged = pd.concat(mut_allele_rep_df_renamed).groupby(nucleotide_ids, as_index=False)

                    # Get counts of each replicate for each allele. In CRISPR-Shrinkage, each allele will be treated as a guide entity 
                    observation_guides = []
                    for index, (name, group) in enumerate(mut_allele_rep_df_renamed_merged):
                        group_noNaN = group.fillna(0)
                        sample_population_raw_count_reps_observation = np.asarray([group_noNaN[enriched_pop_df_reads_colname+"_rep{}".format(rep_i)].sum() for rep_i in reps])
                        control_population_raw_count_reps_observation = np.asarray([group_noNaN[baseline_pop_df_reads_colname+"_rep{}".format(rep_i)].sum() for rep_i in reps])
                        # TODO: Later can add more info to guide, i.e. the allele. But setting the identifer as the df index is good and possibly sufficient.
                        guide = crispr_shrinkage.Guide(identifier="observation_{}".format(index), position=None, sample_population_raw_count_reps=sample_population_raw_count_reps_observation, control_population_raw_count_reps=control_population_raw_count_reps_observation, is_explanatory=True)
                        observation_guides.append(guide)
                        
                    shrinkage_results = crispr_shrinkage.perform_adjustment(
                        negative_control_guides = negative_guides,
                        positive_control_guides = [],
                        observation_guides = observation_guides,
                        num_replicates = len(reps),
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
                    for rep_i in reps:
                        merged_exp_rep_df = exp_merged_rep_df_list[rep_i]
                        wt_allele_df = merged_exp_rep_df[merged_exp_rep_df[nucleotide_ids].sum(axis=1) == 0]

                        # Rename the dataframe to differentiate counts between reps
                        wt_allele_df_renamed = wt_allele_df.rename(columns={enriched_pop_df_reads_colname: enriched_pop_df_reads_colname+"_rep{}".format(rep_i), baseline_pop_df_reads_colname: baseline_pop_df_reads_colname+"_rep{}".format(rep_i)})

                        # Group by allele
                        nucleotide_ids = [col for col in wt_allele_df_renamed.columns if ">" in col]
                        wt_allele_df_merged = wt_allele_df.groupby(nucleotide_ids, as_index=False).sum() # This is for the final dataframe
                        wt_allele_df_renamed_merged = wt_allele_df_renamed.groupby(nucleotide_ids, as_index=False)



                        negative_guides = []
                        for index, (name, group) in enumerate(wt_allele_df_renamed_merged):
                            group_noNaN = group.fillna(0)
                            sample_population_raw_count_reps_observation = np.asarray([group_noNaN[enriched_pop_df_reads_colname+"_rep{}".format(rep_i)].sum()])
                            control_population_raw_count_reps_observation = np.asarray([group_noNaN[baseline_pop_df_reads_colname+"_rep{}".format(rep_i)].sum()])
                            # TODO: Later can add more info to guide, i.e. the allele. But setting the identifer as the df index is good and possibly sufficient.
                            guide = crispr_shrinkage.Guide(identifier="negative_{}".format(index), position=None, sample_population_raw_count_reps=sample_population_raw_count_reps_observation, control_population_raw_count_reps=control_population_raw_count_reps_observation, is_explanatory=True)
                            negative_guides.append(guide)



                        # Get alleles that are mutated
                        mut_allele_df = merged_exp_rep_df[merged_exp_rep_df[nucleotide_ids].sum(axis=1) > 0]

                        # Rename the dataframe to differentiate counts between reps
                        mut_allele_df_renamed = df.rename(columns={enriched_pop_df_reads_colname: enriched_pop_df_reads_colname+"_rep{}".format(rep_i), baseline_pop_df_reads_colname: baseline_pop_df_reads_colname+"_rep{}".format(rep_i)})

                        # Group by allele
                        nucleotide_ids = [col for col in mut_allele_df_renamed.columns if ">" in col]
                        mut_allele_df_merged = mut_allele_df.groupby(nucleotide_ids, as_index=False).sum()
                        mut_allele_df_renamed_merged = mut_allele_df_renamed.groupby(nucleotide_ids, as_index=False)

                        # Get counts of each replicate for each allele. In CRISPR-Shrinkage, each allele will be treated as a guide entity 
                        observation_guides = []
                        for index, (name, group) in enumerate(mut_allele_df_renamed_merged):
                            group_noNaN = group.fillna(0)
                            sample_population_raw_count_reps_observation = np.asarray([group_noNaN[enriched_pop_df_reads_colname+"_rep{}".format(rep_i)].sum()])
                            control_population_raw_count_reps_observation = np.asarray([group_noNaN[baseline_pop_df_reads_colname+"_rep{}".format(rep_i)].sum()])
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
                                                       enriched_pop_df_reads_colname=enriched_pop_df_reads_colname,                               
                                                       baseline_pop_df_reads_colname= baseline_pop_df_reads_colname,
                                                       presort_pop_df_reads_colname=presort_pop_df_reads_colname,
                                                       sigma_scale_normalized= design_matrix_processing_specification.sigma_scale_normalized,
                                                       decay_sigma_scale= design_matrix_processing_specification.decay_sigma_scale,
                                                       K_enriched=design_matrix_processing_specification.K_enriched, 
                                                       K_baseline=design_matrix_processing_specification.K_baseline, 
                                                       a_parameter=design_matrix_processing_specification.a_parameter,
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
                data = merged_experiments_df
            elif experiment_merge_strategy == MillipedeExperimentMergeStrategy.COVARIATE:
                # DEVELOPER NOTE: Ensure that intercept_postfix between per-replicate and per-experiment are different, else there could be overwriting during intercept assignment
                if replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SEPARATE, MillipedeReplicateMergeStrategy.MODELLED_SEPARATE]: # SINGLE MATRIX PER REPLICATE
                    merged_experiment_df_list: List[List[pd.DataFrame]]
                    merged_experiments_df: List[pd.DataFrame]
                    merged_experiments_df = [pd.concat([self.__get_intercept_df(merged_experiment_df_list), pd.concat(merged_experiment_df_i, ignore_index=True)], axis=1) for merged_experiment_df_i in merged_experiment_df_list]
                    merged_experiments_df = [merged_experiments_df_i.fillna(0.0) for merged_experiments_df_i in merged_experiments_df] # TODO 20221021: This is to ensure all intercept values are assigned (since NaNs exist with covariate by experiment) - there is possible if there are other NaN among features that it will be set to 0 unintentionally
                    merged_experiments_df = [__add_supporting_columns_partial(encoding_df = merged_experiments_df_i) for replicate_i, merged_experiments_df_i in enumerate(merged_experiments_df)]
                    #merged_experiments_df = [merged_experiments_df_i[merged_experiments_df_i["total_reads"] > 0] for merged_experiments_df_i in merged_experiments_df] # Ensure non-zero reads to prevent error during modelling
                    
                    data = merged_experiments_df
                elif replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SUM, MillipedeReplicateMergeStrategy.COVARIATE, MillipedeReplicateMergeStrategy.MODELLED_COMBINED]: # SINGLE MATRIX FOR ALL REPLICATES
                    merged_experiment_df_list: List[pd.DataFrame]
                    merged_experiments_df: pd.DataFrame
                    merged_experiments_df = pd.concat([self.__get_intercept_df(merged_experiment_df_list), pd.concat(merged_experiment_df_list, ignore_index=True)], axis=1)
                    merged_experiments_df = merged_experiments_df.fillna(0.0) # TODO 20221021: This is to ensure all intercept values are assigned (since NaNs exist with covariate by experiment) - there is possible if there are other NaN among features that it will be set to 0 unintentionally
                    merged_experiments_df = __add_supporting_columns_partial(encoding_df = merged_experiments_df)
                    #merged_experiments_df = merged_experiments_df[merged_experiments_df["total_reads"] > 0] # Ensure non-zero reads to prevent error during modelling

                    data = merged_experiments_df
            elif experiment_merge_strategy == MillipedeExperimentMergeStrategy.SEPARATE:
                if replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SEPARATE, MillipedeReplicateMergeStrategy.MODELLED_SEPARATE]:
                    merged_experiment_df_list: List[List[pd.DataFrame]]
                    merged_experiment_df_list = [[__add_supporting_columns_partial(encoding_df = merged_rep_df, experiment_i=experiment_i, replicate_i=replicate_i) for replicate_i, merged_rep_df in enumerate(merged_rep_df_list)] for experiment_i, merged_rep_df_list in enumerate(merged_experiment_df_list)]
                    #merged_experiment_df_list = [[merged_rep_df[merged_rep_df["total_reads"] > 0] for merged_rep_df in merged_rep_df_list] for merged_rep_df_list in merged_experiment_df_list] # Ensure non-zero reads to prevent error during modelling

                    data = merged_experiment_df_list
                elif replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SUM, MillipedeReplicateMergeStrategy.COVARIATE, MillipedeReplicateMergeStrategy.MODELLED_COMBINED]:
                    merged_experiment_df_list: List[pd.DataFrame]
                    merged_experiment_df_list = [__add_supporting_columns_partial(encoding_df = merged_reps_df, experiment_i=experiment_i) for experiment_i, merged_reps_df in enumerate(merged_experiment_df_list)]
                    #merged_experiment_df_list = [merged_reps_df[merged_reps_df["total_reads"] > 0] for merged_reps_df in merged_experiment_df_list]

                    data = merged_experiment_df_list
            else:
                raise Exception("Developer error: Unexpected value for MillipedeExperimentMergeStrategy: {}".format(experiment_merge_strategy))

            millipede_input_data: MillipedeInputData = MillipedeInputData(
                data=data,
                data_directory=data_directory, 
                enriched_pop_fn_experiment_list=enriched_pop_fn_experiment_list,
                enriched_pop_df_reads_colname=enriched_pop_df_reads_colname, 
                baseline_pop_fn_experiment_list=baseline_pop_fn_experiment_list,
                baseline_pop_df_reads_colname=baseline_pop_df_reads_colname, 
                presort_pop_fn_experiment_list=presort_pop_fn_experiment_list,
                presort_pop_df_reads_colname=presort_pop_df_reads_colname, 
                reps=reps, 
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

    def __normalize_counts(self,
                          encoding_df: pd.DataFrame,
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

    def __add_supporting_columns(self, 
                                 encoding_df: pd.DataFrame, 
                                 enriched_pop_df_reads_colname: str, 
                                 baseline_pop_df_reads_colname: str,
                                 presort_pop_df_reads_colname: Optional[str],
                                 sigma_scale_normalized: bool,
                                 decay_sigma_scale: bool,
                                 K_enriched: Union[float, List[float], List[List[float]]],
                                 K_baseline: Union[float, List[float], List[List[float]]],
                                 a_parameter: Union[float, List[float], List[List[float]]],
                                 set_offset_as_default: bool,
                                 set_offset_as_total_reads: bool,
                                 set_offset_as_enriched: bool,
                                 set_offset_as_baseline: bool,
                                 set_offset_as_presort: bool,
                                 offset_normalized: bool,
                                 offset_psuedocount: int,
                                 experiment_i: Optional[int] = None,
                                 replicate_i: Optional[int] = None
                                ) -> pd.DataFrame:
        # construct the simplest possible continuous-valued response variable.
        # this response variable is in [-1, 1]
        
        # Get intercept exp and reps for setting :
        intercept_columns = [col for col in  encoding_df.columns if "intercept" in col]
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
                else:
                    exp_index = col.find("exp")
                    if exp_index != -1:
                        intercept_exp_i = int(col[(exp_index + len("exp")):])
                        exp_indices.append(intercept_exp_i)
                        rep_indices.append(replicate_i) # Add the explit replicate ID if only the experiment ID was found in the intercept column
        

        # Original
        enriched_read_counts = encoding_df[enriched_pop_df_reads_colname]
        baseline_read_counts = encoding_df[baseline_pop_df_reads_colname]
        total_read_counts = encoding_df['total_reads']
        
        if 'score' not in encoding_df.columns: 
            encoding_df['score'] = (enriched_read_counts - baseline_read_counts) / (enriched_read_counts + baseline_read_counts) 
            
        # create scale_factor for normal likelihood model
        #if 'scale_factor' not in encoding_df.columns: 
            #encoding_df['scale_factor'] = 1.0 / np.sqrt(encoding_df['total_reads']) # NOTE: Intentionally keeping the total_reads as the raw to avoid being impact by normalization - this could be subject to change
        if 'scale_factor' not in encoding_df.columns:
            def set_scale_factor(input_encoding_df, K_enriched_selected, K_baseline_selected, a_parameter_selected):
                if sigma_scale_normalized:
                    if decay_sigma_scale:
                        input_encoding_df['scale_factor'] = ((decay_function(input_encoding_df[enriched_pop_df_reads_colname], K_enriched_selected, a_parameter_selected))  + (decay_function(input_encoding_df[baseline_pop_df_reads_colname], K_baseline_selected, a_parameter_selected)))/2 
                    else:
                        input_encoding_df['scale_factor'] = (K_enriched_selected / np.sqrt(input_encoding_df[enriched_pop_df_reads_colname])) + (input_encoding_df / np.sqrt(input_encoding_df[baseline_pop_df_reads_colname]))
                else:
                    if decay_sigma_scale:
                        input_encoding_df['scale_factor'] = ((decay_function(input_encoding_df[enriched_pop_df_reads_colname + "_raw"], K_enriched_selected, a_parameter_selected)) + (decay_function(input_encoding_df[baseline_pop_df_reads_colname + "_raw"], K_baseline_selected, a_parameter_selected)))/2 
                    else:
                        input_encoding_df['scale_factor'] = (K_enriched_selected / np.sqrt(input_encoding_df[enriched_pop_df_reads_colname + "_raw"])) + (K_baseline_selected / np.sqrt(input_encoding_df[baseline_pop_df_reads_colname + "_raw"]))
                return input_encoding_df


            def retrieve_sample_parameter(parameter_input, experiment_index, replicate_index):
                if type(parameter_input) is list:
                    assert replicate_index is not None, "Replicate index must be provided"
                    if type(parameter_input[0] is list):
                        assert experiment_index is not None, "Experiment index must be provided"
                        parameter_input_selected = parameter_input[experiment_index][replicate_index]
                    else:
                        parameter_input_selected = parameter_input[rep_index]
                else:
                    parameter_input_selected = parameter_input
                return parameter_input_selected
    
            # If the encoding as intercept columns, then extract the available exp/rep indices, subset by each exp/rep, get the correct sigma_scale parameters, and update the encoding DF
            if intercept_columns:
                # Iterate through each intercept index to get exp/rep index
                sample_encoding_df_list = []
                for intercept_index, intercept_col in enumerate(intercept_columns):
                    exp_index = exp_indices[intercept_index]
                    rep_index = rep_indices[intercept_index]
                    
                    # Get the corresponding sigma scale parameter based on the exp/rep index
                    K_enriched_selected = retrieve_sample_parameter(K_enriched, experiment_index=exp_index, replicate_index=rep_index)
                    K_baseline_selected = retrieve_sample_parameter(K_baseline, experiment_index=exp_index, replicate_index=rep_index)
                    a_parameter_selected = retrieve_sample_parameter(a_parameter, experiment_index=exp_index, replicate_index=rep_index)

                    # Subset the encoding by the intercept index and add scale factor
                    sample_encoding_df = encoding_df[encoding_df[intercept_col] == 1]
                    sample_encoding_df = set_scale_factor(sample_encoding_df, K_enriched_selected=K_enriched_selected, K_baseline_selected=K_baseline_selected, a_parameter_selected=a_parameter_selected)
                    sample_encoding_df_list.append(sample_encoding_df)
                
                # Concatenate all the updated sample encoding DFs into the complete encoding DF
                encoding_df = pd.concat(sample_encoding_df_list, axis=0)
            
            else: # If there are no intercept columns, see if explicit experiment or replicate index is provided
                if replicate_i is None:
                    if experiment_i is None:
                        # If no explicit experiment or replicate index is provided, then expecting a single sigma_scale parameter but must assert the input first
                        assert isinstance(K_enriched, (int, float)), f"K_enriched {K_enriched} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter) must be an int/float type"
                        assert isinstance(K_baseline, (int, float)), f"K_baseline {K_baseline} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter) must be an int/float type"
                        assert isinstance(a_parameter, (int, float)), f"a_parameter {a_parameter} and all sigma_scale_parameters (K_enriched, K_baseline, a_parameter) must be an int/float type"
                        K_enriched_selected = K_enriched
                        K_baseline_selected = K_baseline
                        a_parameter_selected = a_parameter
                else:
                    # If replicate (and experiment) index is provided, get the selected sigma_scale parameters
                    K_enriched_selected = retrieve_sample_parameter(K_enriched, experiment_i, replicate_i)
                    K_baseline_selected = retrieve_sample_parameter(K_baseline, experiment_i, replicate_i)
                    a_parameter_selected = retrieve_sample_parameter(a_parameter, experiment_i, replicate_i)
                
                encoding_df = set_scale_factor(encoding_df, K_enriched_selected=K_enriched_selected, K_baseline_selected=K_baseline_selected, a_parameter_selected=a_parameter_selected)
            
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
  