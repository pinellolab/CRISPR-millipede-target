import numpy as np
import torch
from millipede import NormalLikelihoodVariableSelector
from millipede import BinomialLikelihoodVariableSelector
from millipede import NegativeBinomialLikelihoodVariableSelector
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import crispr_shrinkage

from os.path import exists

from dataclasses import dataclass
from typing import Union, List, Mapping, Tuple, Optional
from functools import partial
from typeguard import typechecked
from enum import Enum
from collections import defaultdict

from .models_inputs import *

# TODO 20221019: Include presort in the filtering, so therefore must also take presort fn as input
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
    wt_normalization: bool
    total_normalization: bool
    sigma_scale_normalized: bool
    K_enriched: float
    K_baseline: float 
                        
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
        print("Passed validation.")
        
        
        __get_data_partial = partial(
            self.__get_data,
            data_directory=self.data_directory, 
            enriched_pop_fn_experiment_list=self.enriched_pop_fn_experiment_list, 
            enriched_pop_df_reads_colname=self.enriched_pop_df_reads_colname, 
            baseline_pop_fn_experiment_list=self.baseline_pop_fn_experiment_list, 
            baseline_pop_df_reads_colname=self.baseline_pop_df_reads_colname, 
            reps=self.reps,
            wt_normalization=self.wt_normalization,
            total_normalization=self.total_normalization,
            sigma_scale_normalized=self.sigma_scale_normalized,
            K_enriched=self.K_enriched,
            K_baseline=self.K_baseline
        )
        # This will be the variable containing the final dictionary with input design matrix for all specifications
        millipede_model_specification_set_with_data: Mapping[str, Tuple[MillipedeModelSpecification, MillipedeInputData]] = dict()
        
        # Helpful note: This retrieves the unique set of input design matrix to generate based on the provided model specifications (specifically the unique set of replicate merge strategy, experiment merge strategy, and cutoff specifications as the model input data only varies based on these criteria)
        millipede_design_matrix_set: Mapping[Tuple[MillipedeReplicateMergeStrategy, MillipedeExperimentMergeStrategy, MillipedeCutoffSpecification, MillipedeShrinkageInput], List[str]] = self.__determine_full_design_matrix_set(self.millipede_model_specification_set)
        self.__millipede_design_matrix_set = millipede_design_matrix_set
        
        # Below generates the input data and assigns to corresponding model specifications
        merge_strategy_and_cutoff_tuple: Tuple[MillipedeReplicateMergeStrategy, MillipedeExperimentMergeStrategy, MillipedeCutoffSpecification, MillipedeShrinkageInput]
        millipede_model_specification_id_list: List[str]
        for merge_strategy_and_cutoff_tuple, millipede_model_specification_id_list in millipede_design_matrix_set.items():
            # Generate input data - most computationally intensive task in this method
            print("Retrieving data for\n\tReplicate Merge Strategy: {} \n\tExperiment Merge Strategy {}\n\tCutoff: {}". format(merge_strategy_and_cutoff_tuple[0], merge_strategy_and_cutoff_tuple[1], merge_strategy_and_cutoff_tuple[2]))
            millipede_input_data: MillipedeInputData = __get_data_partial(
                replicate_merge_strategy=merge_strategy_and_cutoff_tuple[0], 
                experiment_merge_strategy=merge_strategy_and_cutoff_tuple[1],
                cutoff_specification=merge_strategy_and_cutoff_tuple[2],
                shrinkage_input=merge_strategy_and_cutoff_tuple[3]
                
            )
                
            # Assign input data to corresponding model specifications
            for millipede_model_specification_id in millipede_model_specification_id_list:
                millipede_model_specification_set_with_data[millipede_model_specification_id] = (self.millipede_model_specification_set[millipede_model_specification_id], millipede_input_data)
                
        self.millipede_model_specification_set_with_data = millipede_model_specification_set_with_data
                
                

    """ 
    Function to process the encoding dataframe (from encode pipeline script) and create design matrix for milliped
    """
    def __get_data(self, 
                   data_directory: str, 
                   enriched_pop_fn_experiment_list: List[str], 
                   enriched_pop_df_reads_colname: str, 
                   baseline_pop_fn_experiment_list: List[str], 
                   baseline_pop_df_reads_colname: str, 
                   reps: List[int], 
                   replicate_merge_strategy:MillipedeReplicateMergeStrategy, 
                   experiment_merge_strategy:MillipedeExperimentMergeStrategy,
                   cutoff_specification: MillipedeCutoffSpecification,
                   shrinkage_input: Union[MillipedeShrinkageInput, None],
                   wt_normalization: bool,
                   total_normalization: bool,
                   sigma_scale_normalized: bool,
                   K_enriched: float,
                   K_baseline: float) -> MillipedeInputData:
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            '''
                Input validation
            '''
            if experiment_merge_strategy == MillipedeExperimentMergeStrategy.SUM:
                assert replicate_merge_strategy == MillipedeReplicateMergeStrategy.SUM, "replicate_merge_strategy must be SUM if experiment_merge_strategy is SUM"
            

            #
            # Process the replicate dataframes:
            #
            merged_experiment_df_list: Union[List[pd.DataFrame], List[List[pd.DataFrame]]] = []
            # Iterate through the experiments
            for experiment_index in range(len(enriched_pop_fn_experiment_list)):
                # Get the enriched_population and baseline_population for the experiment
                enriched_pop_exp_fn = enriched_pop_fn_experiment_list[experiment_index]
                baseline_pop_exp_fn = baseline_pop_fn_experiment_list[experiment_index]
            
            
                # Iterate through each replicate of the experiment
                exp_merged_rep_df_list = []
                for rep in reps:
                    '''
                        Check file directories
                    '''
                    enriched_pop_full_fn_exp_rep = (data_directory + '/' + enriched_pop_exp_fn).format(rep)
                    baseline_pop_full_fn_exp_rep = (data_directory + '/' + baseline_pop_exp_fn).format(rep)
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
                    nucleotide_ids = enriched_pop_nt_columns

                    # Concat the enriched and baseline population dataframes together
                    merged_exp_rep_df: pd.DataFrame = pd.concat([enriched_pop_exp_rep_df, baseline_pop_exp_rep_df]).groupby(nucleotide_ids, as_index=False).sum()

                    # filter based on the per_replicate_each_condition_num_cutoff
                    merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df[baseline_pop_df_reads_colname] >= cutoff_specification.per_replicate_each_condition_num_cutoff]
                    merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df[enriched_pop_df_reads_colname] >= cutoff_specification.per_replicate_each_condition_num_cutoff]
                    merged_exp_rep_df['total_reads'] = merged_exp_rep_df[baseline_pop_df_reads_colname] + merged_exp_rep_df[enriched_pop_df_reads_colname]

                    # filter on total reads based on the per_replicate_all_condition_num_cutoff
                    total_alleles_pre_filter = merged_exp_rep_df.values.shape[0]
                    merged_exp_rep_df = merged_exp_rep_df[merged_exp_rep_df["total_reads"] >= cutoff_specification.per_replicate_all_condition_num_cutoff]
                    
                    # Normalize counts
                    merged_exp_rep_normalized_df = self.__normalize_counts(merged_exp_rep_df, enriched_pop_df_reads_colname, baseline_pop_df_reads_colname, nucleotide_ids, wt_normalization, total_normalization)
                    
                    # Add to the replicate list
                    exp_merged_rep_df_list.append(merged_exp_rep_normalized_df)
                    
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
                                                       sigma_scale_normalized= sigma_scale_normalized,
                                                       K_enriched=K_enriched,
                                                       K_baseline=K_baseline
                                                      )
            
            data = None
            if experiment_merge_strategy == MillipedeExperimentMergeStrategy.SUM:
                nucleotide_ids = [col for col in merged_experiment_df_list[0].columns if ">" in col]
                merged_experiments_df: pd.DataFrame
                merged_experiments_df = pd.concat(merged_experiment_df_list).groupby(nucleotide_ids, as_index=False).sum()
                merged_experiments_df = merged_experiments_df[merged_experiments_df["total_reads"] >= cutoff_specification.all_experiment_num_cutoff]
                merged_experiments_df = merged_experiments_df[merged_experiments_df["total_reads"] > 0] # Ensure non-zero reads to prevent error during modelling
                merged_experiments_df = __add_supporting_columns_partial(encoding_df = merged_experiments_df)
                data = merged_experiments_df
            elif experiment_merge_strategy == MillipedeExperimentMergeStrategy.COVARIATE:
                # DEVELOPER NOTE: Ensure that intercept_postfix between per-replicate and per-experiment are different, else there could be overwriting during intercept assignment
                if replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SEPARATE, MillipedeReplicateMergeStrategy.MODELLED_SEPARATE]: # SINGLE MATRIX PER REPLICATE
                    merged_experiment_df_list: List[List[pd.DataFrame]]
                    merged_experiments_df: List[pd.DataFrame]
                    merged_experiments_df = [pd.concat([self.__get_intercept_df(merged_experiment_df_list), pd.concat(merged_experiment_df_i, ignore_index=True)], axis=1) for merged_experiment_df_i in merged_experiment_df_list]
                    merged_experiments_df = [merged_experiments_df_i.fillna(0.0) for merged_experiments_df_i in merged_experiments_df] # TODO 20221021: This is to ensure all intercept values are assigned (since NaNs exist with covariate by experiment) - there is possible if there are other NaN among features that it will be set to 0 unintentionally
                    merged_experiments_df = [__add_supporting_columns_partial(encoding_df = merged_experiments_df_i) for merged_experiments_df_i in merged_experiments_df]
                    merged_experiments_df = [merged_experiments_df_i[merged_experiments_df_i["total_reads"] > 0] for merged_experiments_df_i in merged_experiments_df] # Ensure non-zero reads to prevent error during modelling
                    data = merged_experiments_df
                elif replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SUM, MillipedeReplicateMergeStrategy.COVARIATE, MillipedeReplicateMergeStrategy.MODELLED_COMBINED]: # SINGLE MATRIX FOR ALL REPLICATES
                    merged_experiment_df_list: List[pd.DataFrame]
                    merged_experiments_df: pd.DataFrame
                    merged_experiments_df = pd.concat([self.__get_intercept_df(merged_experiment_df_list), pd.concat(merged_experiment_df_list, ignore_index=True)], axis=1)
                    merged_experiments_df = merged_experiments_df.fillna(0.0) # TODO 20221021: This is to ensure all intercept values are assigned (since NaNs exist with covariate by experiment) - there is possible if there are other NaN among features that it will be set to 0 unintentionally
                    merged_experiments_df = __add_supporting_columns_partial(encoding_df = merged_experiments_df)
                    merged_experiments_df = merged_experiments_df[merged_experiments_df["total_reads"] > 0] # Ensure non-zero reads to prevent error during modelling
                    data = merged_experiments_df
                    
            elif experiment_merge_strategy == MillipedeExperimentMergeStrategy.SEPARATE:
                if replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SEPARATE, MillipedeReplicateMergeStrategy.MODELLED_SEPARATE]:
                    merged_experiment_df_list: List[List[pd.DataFrame]]
                    merged_experiment_df_list = [[__add_supporting_columns_partial(encoding_df = merged_rep_df) for merged_rep_df in merged_rep_df_list] for merged_rep_df_list in merged_experiment_df_list]
                    merged_experiment_df_list = [[merged_rep_df[merged_rep_df["total_reads"] > 0] for merged_rep_df in merged_rep_df_list] for merged_rep_df_list in merged_experiment_df_list] # Ensure non-zero reads to prevent error during modelling
                    data = merged_experiment_df_list
                elif replicate_merge_strategy in [MillipedeReplicateMergeStrategy.SUM, MillipedeReplicateMergeStrategy.COVARIATE, MillipedeReplicateMergeStrategy.MODELLED_COMBINED]:
                    merged_experiment_df_list: List[pd.DataFrame]
                    merged_experiment_df_list = [__add_supporting_columns_partial(encoding_df = merged_reps_df) for merged_reps_df in merged_experiment_df_list]
                    merged_experiment_df_list = [merged_reps_df[merged_reps_df["total_reads"] > 0] for merged_reps_df in merged_experiment_df_list]
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
                reps=reps, 
                replicate_merge_strategy=replicate_merge_strategy, 
                experiment_merge_strategy=experiment_merge_strategy,
                cutoff_specification=cutoff_specification
            )
            
            return millipede_input_data

    def __determine_full_design_matrix_set(self, millipede_model_specification_set: Mapping[str, MillipedeModelSpecification]) -> Mapping[Tuple[MillipedeReplicateMergeStrategy, MillipedeExperimentMergeStrategy, MillipedeCutoffSpecification, MillipedeShrinkageInput], List[str]]:
        """
            This determines what set of design matrices to generate based on the set of Millipede model specifications - this is determined based on the replicate/experiment merge strategies

            The returned variable is a dictionary with the replicate/experiment merge strategy tuple as the key and the list of Millipede model specifications IDs as the value to ensure the model specification that each design matrix maps to.
        """
        millipede_design_matrix_set: Mapping[Tuple[MillipedeReplicateMergeStrategy, MillipedeExperimentMergeStrategy, MillipedeCutoffSpecification, MillipedeShrinkageInput], List[str]] = defaultdict(list)

        millipede_model_specification_id: str
        millipede_model_specification: MillipedeModelSpecification
        for millipede_model_specification_id, millipede_model_specification in millipede_model_specification_set.items():
            millipede_design_matrix_set[(millipede_model_specification.replicate_merge_strategy, millipede_model_specification.experiment_merge_strategy, millipede_model_specification.cutoff_specification, millipede_model_specification.shrinkage_input)].append(millipede_model_specification_id)

        return millipede_design_matrix_set

    def __normalize_counts(self,
                          encoding_df: pd.DataFrame,
                          enriched_pop_df_reads_colname: str,
                          baseline_pop_df_reads_colname: str,
                          nucleotide_ids: List[str],
                          wt_normalization: bool,
                          total_normalization: bool) -> pd.DataFrame:
        # TODO 5/15/23: Normalization is set to True always! Make it an input variable. Also, it should directly change the count rather than just the score
        # TODO 5/15/23: Also, allow normalization either by library size or by WT reads. For now, will just do WT reads
        
        # Original
        enriched_read_counts = encoding_df[enriched_pop_df_reads_colname]
        baseline_read_counts = encoding_df[baseline_pop_df_reads_colname]
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
        else:
            encoding_df[enriched_pop_df_reads_colname + "_raw"] = encoding_df[enriched_pop_df_reads_colname]
            encoding_df[baseline_pop_df_reads_colname + "_raw"] = encoding_df[baseline_pop_df_reads_colname]
        
        return encoding_df

    def __add_supporting_columns(self, 
                                 encoding_df: pd.DataFrame, 
                                 enriched_pop_df_reads_colname: str, 
                                 baseline_pop_df_reads_colname: str,
                                 sigma_scale_normalized: bool,
                                 K_enriched: float,
                                 K_baseline: float
                                ) -> pd.DataFrame:
        # construct the simplest possible continuous-valued response variable.
        # this response variable is in [-1, 1]
        
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
            if sigma_scale_normalized:
                encoding_df['scale_factor'] = (K_enriched / np.sqrt(encoding_df[enriched_pop_df_reads_colname])) + (K_baseline / np.sqrt(encoding_df[baseline_pop_df_reads_colname]))  # NOTE: Intentionally keeping the total_reads as the raw to avoid being impact by normalization - this could be subject to change
            else:
                encoding_df['scale_factor'] = (K_enriched / np.sqrt(encoding_df[enriched_pop_df_reads_colname + "_raw"])) + (K_baseline / np.sqrt(encoding_df[baseline_pop_df_reads_colname + "_raw"]))  # NOTE: Intentionally keeping the total_reads as the raw to avoid being impact by normalization - this could be subject to change
                        
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
            
            # Iterate through the data depending on its list structure, run models, and add to set
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
        models: Mapping[MillipedeModelType, Union[NormalLikelihoodVariableSelector, BinomialLikelihoodVariableSelector]] = {}
        for i, model_type in enumerate(millipede_model_specification.model_types):
            if model_type == MillipedeModelType.NORMAL:
                print("Preparing data for model {}, {}/{}".format(model_type.value, i+1, len(model_types)))
                required_columns = intercept_columns + nucleotide_ids + ['score', 'scale_factor'] 
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
    def set_variables_constructor(self, enriched_pop_encodings_df_list: List[pd.DataFrame], baseline_pop_encodings_df_list: List[pd.DataFrame], presort_pop_encodings_df_list: Optional[List[pd.DataFrame]] = None, wt_pop_encodings_df_list: Optional[List[pd.DataFrame]] = None):
        self.enriched_pop_encodings_df_list = self.enriched_pop_encodings_df_list
        self.baseline_pop_encodings_df_list = self.baseline_pop_encodings_df_list
        self.presort_pop_encodings_df_list = self.presort_pop_encodings_df_list
        self.wt_pop_encodings_df_list = self.wt_pop_encodings_df_list
        
        self.__post_validate()
        
        return self
        
    def read_in_files_constructor(self, 
                                  enriched_pop_fn_encodings_experiment_list: List[str], 
                                  baseline_pop_fn_encodings_experiment_list: List[str], 
                                  experiment_labels: List[str],
                                  presort_pop_fn_encodings_experiment_list: Optional[List[str]] = None,
                                  ctrl_pop_fn_encodings: Optional[Union[list, str]] = None,
                                  ctrl_pop_labels: Optional[Union[list, str]]=None,
                                  reps:List[int]=None):
        self.enriched_pop_fn_encodings_experiment_list = enriched_pop_fn_encodings_experiment_list
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
        for fn in enriched_pop_fn_encodings_experiment_list:
            assert "{}" in fn, "Filename must have '{}' to replace with replicate ID, provided filename: " + str(fn)
            self.__check_file_locations(fn, reps)
        for fn in baseline_pop_fn_encodings_experiment_list:
            assert "{}" in fn, "Filename must have '{}' to replace with replicate ID, provided filename: " + str(fn)
            self.__check_file_locations(fn, reps)
        if presort_pop_fn_encodings_experiment_list != None: 
            for fn in presort_pop_fn_encodings_experiment_list:
                assert "{}" in fn, "Filename must have '{}' to replace with replicate ID, provided filename: " + str(fn)  
                self.__check_file_locations(fn, reps)

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
        def read_encodings_in_nested_list(sup_encoding_fn: Union[list, str], reps: Optional[List[int]]=None, _depth:int=0, _breadth:int=0):
            if isinstance(sup_encoding_fn, list):
                sup_encoding_df_list = []
                for i, subb_encoding_fn in enumerate(sup_encoding_fn):
                    sup_encoding_df_list.append(read_encodings_in_nested_list(subb_encoding_fn, reps, _depth=_depth+1, _breadth=i))
                return sup_encoding_df_list
            elif isinstance(sup_encoding_fn, str):
                if reps != None:
                    sup_encoding_df_reps_list = []
                    for rep in reps:
                        try:
                            sup_encoding_df_reps_list.append(pd.read_pickle(sup_encoding_fn.format(rep)))
                        except Exception as e:
                            raise Exception("Error reading encoding {} in provided position of filename list (depth={}, breadth={}); original exception: {}".format(subb_encoding_fn.format(rep, depth, breadth, str(e))))
                    return sup_encoding_df_reps_list
                else:
                    try:
                        return pd.read_pickle(sup_encoding_fn)
                    except Exception as e:
                        raise Exception("Error reading encoding {} in provided position of filename list (depth={}, breadth={}); original exception: {}".format(subb_encoding_fn.format(rep, depth, breadth, str(e))))

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
        
    def __check_file_locations(self, fn: str, reps: Optional[List[int]]=None):
        if reps != None:
            for rep in reps:
                assert exists(fn.format(rep)), "File not found: " + fn.format(rep)
        else:
            assert exists(fn), "File not found: " + fn

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
        
        def generate_per_position_editing_frequency_for_ctrl(sup_ctrl_pop_encodings_df_list: Union[list, pd.DataFrame], editing_frequency_callable: Callable, _depth:int=0, _breadth:int=0) -> Union[list, pd.Series]:
            if isinstance(sup_ctrl_pop_encodings_df_list, list):
                sup_ctrl_pop_encoding_editing_freq_list = []
                for i, sup_ctrl_pop_encodings_df in enumerate(sup_ctrl_pop_encodings_df_list):
                    sup_ctrl_pop_encoding_editing_freq_list.append(generate_per_position_editing_frequency_for_ctrl(sup_ctrl_pop_encodings_df, _depth=_depth+1, _breadth=i))
                return sup_ctrl_pop_encoding_editing_freq_list
            elif isinstance(sup_ctrl_pop_encodings_df_list, pd.DataFrame):
                return editing_frequency_callable(sup_ctrl_pop_encodings_df_list)
         
        
        
        self.ctrl_pop_encoding_editing_freq_list = generate_per_position_editing_frequency_for_ctrl(self.raw_encodings.ctrl_pop_encodings_df_list, editing_frequency_callable=self.__generate_per_position_editing_frequency)
        self.ctrl_pop_encoding_editing_per_variant_freq_list = generate_per_position_editing_frequency_for_ctrl(self.raw_encodings.ctrl_pop_encodings_df_list, editing_frequency_callable=self.__generate_per_variant_editing_frequency)
        
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
            
        # TODO: This function needs to be tested on several use cases varying the structure of  sup_ctrl_pop_encoding_editing_freq (note 10/22/2022)
        def calculate_ctrl_pop_encoding_editing_freq_avg(sup_ctrl_pop_encoding_editing_freq: Union[list, pd.Series], ctrl_pop_encoding_editing_freq_avg_level:int, _depth:int=0, _breadth:int=0) -> Union[list, pd.Series]:
            if isinstance(sup_ctrl_pop_encoding_editing_freq, list):
                sup_ctrl_pop_encoding_editing_freq_list = []
                for i, subb_ctrl_pop_encoding_editing_freq in enumerate(sup_ctrl_pop_encoding_editing_freq_list):
                    sup_ctrl_pop_encoding_editing_freq_list.append(calculate_ctrl_pop_encoding_editing_freq_avg(subb_ctrl_pop_encoding_editing_freq, ctrl_pop_encoding_editing_freq_avg_level, _depth=_depth+1, _breadth=i))
                
                # TODO: This function would likely not work if depths are different between different entries in list - would need to implement even a more dynamic way of calculating average - added assertion below to check for this (note 10/22/2022)
                assert len(sup_ctrl_pop_encoding_editing_freq_list) > 0, "Failed due to empty list in raw_encodings.ctrl_pop_encodings_df_list"
                if isinstance(sup_ctrl_pop_encoding_editing_freq_list[0], pd.Series):
                    for subb_ctrl_pop_encoding_editing_freq_list in sup_ctrl_pop_encoding_editing_freq_list:
                        assert isinstance(subb_ctrl_pop_encoding_editing_freq_list, pd.Series), "raw_encodings.ctrl_pop_encodings_df_list must be perfectly height balanced, if your control sample design is inherently not height balanced, contact the developers to modify package"
                    # Do nothing, already in correct format (likely the second-to-last depth level, or the average was calculated a level down)
                else:
                    sup_ctrl_pop_encoding_editing_freq_list = [subb_subb_ctrl_pop_encoding_editing_freq for subb_ctrl_pop_encoding_editing_freq in sup_ctrl_pop_encoding_editing_freq_list for subb_subb_ctrl_pop_encoding_editing_freq in subb_ctrl_pop_encoding_editing_freq]
                
                if _depth == ctrl_pop_encoding_editing_freq_avg_level:
                    # Depth is at the level of the specified level, thus calculate the average
                    sup_ctrl_pop_encoding_editing_freq_avg = sum(sup_ctrl_pop_encoding_editing_freq_list) / len(sup_ctrl_pop_encoding_editing_freq_list)
                    return sup_ctrl_pop_encoding_editing_freq_avg
                return 
            elif isinstance(sup_ctrl_pop_encoding_editing_freq, pd.Series):
                if _depth == ctrl_pop_encoding_editing_freq_avg_level:
                    print("ctrl_pop_encoding_editing_freq_avg_level was set to max depth of ctrl_pop_encodings_df_list, so no average was calculated")
                    return sup_ctrl_pop_encoding_editing_freq
                elif _depth > ctrl_pop_encoding_editing_freq_avg_level:
                    # Nothing to do but pass up the freq series
                    return sup_ctrl_pop_encoding_editing_freq
                elif _depth < ctrl_pop_encoding_editing_freq_avg_level:
                    raise Exception("ctrl_pop_encoding_editing_freq_avg_level {} was set higher than max depth of ctrl_pop_encodings_df_list {}, must be lower or equal to max depth".format(ctrl_pop_encoding_editing_freq_avg_level, _depth))
        
        # TODO: This function needs to be tested on several use cases varying the structure of  sup_ctrl_pop_encoding_editing_freq (note 10/22/2022)
        def get_max_depth_of_ctrl_pop_encoding_editing_freq(sup_ctrl_pop_encoding_editing_freq: Union[list, pd.Series], _depth:int=0, _breadth:int=0) -> int:
            if isinstance(sup_ctrl_pop_encoding_editing_freq, list):
                sup_ctrl_pop_encoding_editing_freq_list = []
                for i, subb_ctrl_pop_encoding_editing_freq in enumerate(sup_ctrl_pop_encoding_editing_freq_list):
                    sup_ctrl_pop_encoding_editing_freq_list.append(calculate_ctrl_pop_encoding_editing_freq_avg(subb_ctrl_pop_encoding_editing_freq, ctrl_pop_encoding_editing_freq_avg_level, _depth=_depth+1, _breadth=i))
                return max(sup_ctrl_pop_encoding_editing_freq_list)
            elif isinstance(sup_ctrl_pop_encoding_editing_freq, pd.Series):
                return _depth
        
        if hasattr(self, "ctrl_pop_encoding_editing_freq_list"):
            max_depth: int=get_max_depth_of_ctrl_pop_encoding_editing_freq(self.ctrl_pop_encoding_editing_freq_list)
            
            ctrl_pop_encoding_editing_freq_avg: Mapping[int, Union[list, pd.Series]] = {}
            for level in range(max_depth+1):
                ctrl_pop_encoding_editing_freq_avg[level] = calculate_ctrl_pop_encoding_editing_freq_avg(self.ctrl_pop_encoding_editing_freq_list, ctrl_pop_encoding_editing_freq_avg_level=level)
            self.ctrl_pop_encoding_editing_freq_avg = ctrl_pop_encoding_editing_freq_avg
            
            ctrl_pop_encoding_editing_per_variant_freq_avg: Mapping[int, Union[list, pd.Series]] = {}
            for level in range(max_depth+1):
                ctrl_pop_encoding_editing_per_variant_freq_avg[level] = calculate_ctrl_pop_encoding_editing_freq_avg(self.ctrl_pop_encoding_editing_per_variant_freq_list, ctrl_pop_encoding_editing_freq_avg_level=level)
            self.ctrl_pop_encoding_editing_per_variant_freq_avg = ctrl_pop_encoding_editing_per_variant_freq_avg
            
        self.__validated = True
    
    def __generate_per_position_editing_frequency(self, encoding_df: pd.DataFrame) -> pd.Series:
        encoding_df_position_collapsed = pd.DataFrame([encoding_df.iloc[:, encoding_df.columns.get_level_values("Position") == position].sum(axis=1)>0 for position in encoding_df.columns.levels[1]]).T
        encoding_df_position_collapsed_freq = (encoding_df_position_collapsed.sum(axis=0)/encoding_df_position_collapsed.shape[0])
        return encoding_df_position_collapsed_freq
    
    def __generate_per_variant_editing_frequency(self, encoding_df: pd.DataFrame) -> pd.Series:
        encoding_df_freq = (encoding_df.sum(axis=0)/encoding_df.shape[0])
        return encoding_df_freq

