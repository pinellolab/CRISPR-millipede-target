import numpy as np
import torch
from millipede import NormalLikelihoodVariableSelector
from millipede import BinomialLikelihoodVariableSelector
from millipede import NegativeBinomialLikelihoodVariableSelector
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import crispr_shrinkage

import pickle
from datetime import date

from os import listdir
from os.path import isfile, join

'''
    Save a pickle for caching that is notated by the date
'''
def save_or_load_pickle(directory, label, py_object = None, date_string = None):
    if date_string == None:
        today = date.today()
        date_string = str(today.year) + ("0" + str(today.month) if today.month < 10 else str(today.month)) + str(today.day)
    
    filename = directory + label + "_" + date_string + '.pickle'
    print(filename)
    if py_object == None:
        with open(filename, 'rb') as handle:
            py_object = pickle.load(handle)
            return py_object
    else:
        with open(filename, 'wb') as handle:
            pickle.dump(py_object, handle, protocol=pickle.HIGHEST_PROTOCOL)     

'''
    Retrieve all pickles with a label, specifically to identify versions available
'''
def display_all_pickle_versions(directory, label):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and label == f[:len(label)]]


from os.path import exists

""" 
Function to process the encoding dataframe (from encode pipeline script) and create design matrix for milliped
"""
def get_data(data_dir, high_filename, low_filename, keep_score = True, keep_scale_factor=False, keep_total=False, keep_cd19pos=False, rep=1,               # which replicate
             num_cutoff=2,       # minimum total reads for each allele
             kmer=1              # controls feature representation
             ):
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        assert num_cutoff > 1
        assert kmer in [1, 2]

        

        f = data_dir + '/' + high_filename
        high = pd.read_csv(f.format(rep), sep='\t').fillna(value=0.0)
        high_nt_columns = [col for col in high.columns if ">" in col]
        high = high[high_nt_columns + ['#Reads_CD19minus']]

        f = data_dir + '/' + low_filename
        low = pd.read_csv(f.format(rep), sep='\t').fillna(value=0.0)
        low_nt_columns = [col for col in low.columns if ">" in col]
        low = low[low_nt_columns + ['#Reads_CD19plus']]

        merged = pd.concat([high, low]).groupby(high_nt_columns, as_index=False).sum()
        merged.rename(inplace=True, columns={'#Reads_CD19plus': 'cd19pos', '#Reads_CD19minus': 'cd19neg'})

        nucleotide_ids = high_nt_columns

        # remove alleles that have low==0 or high==0
        merged = merged[merged.cd19pos > 0]
        merged = merged[merged.cd19neg > 0]
        merged['total_reads'] = merged['cd19pos'] + merged['cd19neg']

        # construct the simplest possible continuous-valued response variable.
        # this response variable is in [-1, 1]
        merged['score'] = (merged['cd19pos'] - merged['cd19neg']) / merged['total_reads']

        # create scale_factor for normal likelihood model
        merged['scale_factor'] = 1.0 / np.sqrt(merged['total_reads'])
        
        # filter on total reads
        total_alleles_pre_filter = merged.values.shape[0]
        merged = merged[merged.total_reads >= num_cutoff]
        
        kept_columns = []
        if keep_score:
            kept_columns.append('score')
            
        if keep_total:
            kept_columns.append("total_reads")
        
        if keep_cd19pos:
            kept_columns.append("cd19pos")
            
        if keep_scale_factor:
            kept_columns.append("scale_factor")
        
        merged = merged[high_nt_columns + kept_columns]    
        print("Using num_cutoff of {} and keeping {} of {} total alleles".format(num_cutoff, merged.values.shape[0],
                                                                                 total_alleles_pre_filter))
        return merged
    
    
def get_data_merged(data_dir, high_filename, low_filename, keep_score = True, keep_scale_factor=False, keep_total=False, keep_cd19pos=False,               # which replicate
             num_cutoff=2,       # minimum total reads for each allele
             kmer=1,              # controls feature representation
            reps = [1,2,3]
             ):
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        assert num_cutoff > 1
        assert kmer in [1, 2]

        merged_list = []
        for rep in reps:
            f = data_dir + '/' + high_filename
            high = pd.read_csv(f.format(rep), sep='\t').fillna(value=0.0)
            high_nt_columns = [col for col in high.columns if ">" in col]
            high = high[high_nt_columns + ['#Reads_CD19minus']]

            f = data_dir + '/' + low_filename
            low = pd.read_csv(f.format(rep), sep='\t').fillna(value=0.0)
            low_nt_columns = [col for col in low.columns if ">" in col]
            low = low[low_nt_columns + ['#Reads_CD19plus']]

            merged = pd.concat([high, low]).groupby(high_nt_columns, as_index=False).sum()
            merged.rename(inplace=True, columns={'#Reads_CD19plus': 'cd19pos', '#Reads_CD19minus': 'cd19neg'})

            merged_list.append(merged)
        
        nt_columns = [col for col in merged_list[0].columns if ">" in col] 
        merged_reps = pd.concat(merged_list).groupby(nt_columns, as_index=False).sum()
        
        # remove alleles that have low==0 or high==0
        merged_reps = merged_reps[merged_reps.cd19pos > 0]
        merged_reps = merged_reps[merged_reps.cd19neg > 0]
        merged_reps['total_reads'] = merged_reps['cd19pos'] + merged_reps['cd19neg']

        # construct the simplest possible continuous-valued response variable.
        # this response variable is in [-1, 1]
        merged_reps['score'] = (merged_reps['cd19pos'] - merged_reps['cd19neg']) / merged_reps['total_reads']

        # create scale_factor for normal likelihood model
        merged_reps['scale_factor'] = 1.0 / np.sqrt(merged_reps['total_reads'])
        
        # filter on total reads
        total_alleles_pre_filter = merged_reps.values.shape[0]
        merged_reps = merged_reps[merged_reps.total_reads >= num_cutoff]
        
        kept_columns = []
        if keep_score:
            kept_columns.append('score')
        if keep_total:
            kept_columns.append("total_reads")
        
        if keep_cd19pos:
            kept_columns.append("cd19pos")
            
        if keep_scale_factor:
            kept_columns.append("scale_factor")
            
        merged_reps = merged_reps[high_nt_columns + kept_columns]    
        print("Using num_cutoff of {} and keeping {} of {} total alleles".format(num_cutoff, merged_reps.values.shape[0], total_alleles_pre_filter))
        return merged_reps
    
    
def get_intercept_df(dataframe_list, reps):
    intercept_list = []
    for intercept_i, _ in enumerate(reps):
        intercept_i_list = []
        for rep_i, _ in enumerate(reps):
            if intercept_i == rep_i:
                intercept_i_list.extend(np.ones(dataframe_list[rep_i].shape[0]))
            else:
                intercept_i_list.extend(np.zeros(dataframe_list[rep_i].shape[0]))
        intercept_list.append(intercept_i_list)

    intercept_df = pd.DataFrame(intercept_list).transpose()
    intercept_df.columns = ["intercept"  + str(rep) for rep in reps]
    return intercept_df

from dataclasses import dataclass
from typing import Union, List, Tuple

@dataclass
class MillipedeShrinkageInput:
    #negative_control_guides: List[Guide],
    #positive_control_guides: List[Guide],
    #observation_guides: List[Guide],
    #num_replicates: int,
    sample_population_scaling_factors_exp_list: Union[List[float], None] 
    control_population_scaling_factors_exp_list: Union[List[float], None]
    deviation_weights: Union[List[float], None]
    KL_guide_set_weights: Union[List[float], None] 
    include_observational_guides_in_fit: bool = True
    include_positive_control_guides_in_fit: bool = False
    singleton_imputation_prior_strength: Union[List[float], None] = None
    shrinkage_prior_strength: Union[List[float], None] = None
    monte_carlo_trials: int = 1000
    posterior_estimator: str  = "mean"
    LFC_rescaled_null_interval: Tuple[float,float] = None
    LFC_null_interval: Tuple[float,float]  = None
    LFC_rep_rescaled_null_interval: List[Tuple[float,float]]  = None
    LFC_rep_null_interval: List[Tuple[float,float]]  = None
    null_proportion: Tuple[float, float]  = None
    random_seed: Union[int, None]  = None
    cores: int  = 1
    
    def __hash__(self):
        # Create a list of attribute values
        attr_values = [value if not isinstance(value, np.ndarray) else value.tobytes() for value in self.__dict__.values()]
        # Create a tuple of attribute values
        attr_tuple = tuple(attr_values)
        # Return the hash value of the tuple
        return hash(attr_tuple)
    

from os.path import exists
from typing import Union, List, Mapping, Tuple, Optional
from functools import partial
from typeguard import typechecked
from enum import Enum
from dataclasses import dataclass

# TODO: Implement support for technical replicates (perhaps rename below to BiologicalReplicate, then add TechnicalReplicate merge strategy) note 10/22/2022
class MillipedeTechnicalReplicateMergeStrategy(Enum):
    SEPARATE = "SEPARATE"
    SUM = "SUM"
    COVARIATE = "COVARIATE"

class MillipedeReplicateMergeStrategy(Enum):
    """
        Defines how separate replicates will be treated during modelling
    """
    SEPARATE = "SEPARATE"
    SUM = "SUM"
    COVARIATE = "COVARIATE"
    MODELLED_SEPARATE = "MODELLED_SEPARATE"
    MODELLED_COMBINED = "MODELLED_COMBINED"
    
class MillipedeExperimentMergeStrategy(Enum):
    """
        Defines how separate experiments will be treated during modelling
    """
    SEPARATE = "SEPARATE"
    SUM = "SUM"
    COVARIATE = "COVARIATE"

class MillipedeModelType(Enum):
    """
        Defines the Millipede model likelihood function used
    """
    NORMAL = "NORMAL"
    NORMAL_SIGMA_SCALED = "NORMAL_SIGMA_SCALED"
    BINOMIAL = "BINOMIAL"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"

# Deprecated - kmer is only for the nuclease experiments - in the future, whether the encoding was for nuclease or for base-editing can be specified. Later implement nuclease screen support
class MillipedeKmer(Enum):
    """
        Determine whether Millipede model features will be singleton or doublets
    """
    ONE = 1
    TWO = 2
    
@dataclass
class MillipedeCutoffSpecification():
    per_replicate_each_condition_num_cutoff:int = 0
    per_replicate_all_condition_num_cutoff:int = 0 
    all_replicate_num_cutoff:int = 2
    all_experiment_num_cutoff:int = 0
        
    def __hash__(self):
        return hash((self.per_replicate_each_condition_num_cutoff, self.per_replicate_all_condition_num_cutoff, self.all_replicate_num_cutoff, self.all_experiment_num_cutoff))
    
    def __str__(self):
        return "per_replicate_each_condition_num_cutoff={};per_replicate_all_condition_num_cutoff={};all_replicate_num_cutoff={};all_experiment_num_cutoff={}".format(self.per_replicate_each_condition_num_cutoff, self.per_replicate_all_condition_num_cutoff, self.all_replicate_num_cutoff, self.all_experiment_num_cutoff)
    
    def __repr__(self):
        return str(self)

@dataclass
class MillipedeModelSpecification:
    """
        Defines all specifications to produce Millipede model(s)
    """
    model_types: List[MillipedeModelType]
    replicate_merge_strategy: MillipedeReplicateMergeStrategy
    experiment_merge_strategy: MillipedeExperimentMergeStrategy
    cutoff_specification: MillipedeCutoffSpecification
    shrinkage_input: Union[MillipedeShrinkageInput, None] = None
    
    def validate_merge_strategies(self, replicate_merge_strategy: MillipedeReplicateMergeStrategy, experiment_merge_strategy:MillipedeExperimentMergeStrategy):
        if experiment_merge_strategy == MillipedeExperimentMergeStrategy.SUM:
                assert replicate_merge_strategy == MillipedeReplicateMergeStrategy.SUM, "replicate_merge_strategy must be SUM if experiment_merge_strategy is SUM"
                
    # NOTE 20231219: While updating the code to get it working for sg219, the methods below throw an error. Lookup how to create methods in dataclasses
    def __post_init__(self):
        self.validate_merge_strategies(self.replicate_merge_strategy, self.experiment_merge_strategy)
        
    


# TODO (note 10/23/2022): Convert to dataclass or pydantic
@dataclass
class MillipedeInputData:
    """
        Provides relevant input data for a Millipede model specification
    """
    data: Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]]
    data_directory: str
    enriched_pop_fn_experiment_list: List[str]
    enriched_pop_df_reads_colname: str
    baseline_pop_fn_experiment_list: List[str]
    baseline_pop_df_reads_colname: str
    reps: List[int]
    cutoff_specification: MillipedeCutoffSpecification
    replicate_merge_strategy:MillipedeReplicateMergeStrategy
    experiment_merge_strategy:MillipedeExperimentMergeStrategy

from collections import defaultdict

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
            reps=self.reps
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
                   shrinkage_input: Union[MillipedeShrinkageInput, None]) -> MillipedeInputData:
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            '''
                Input validation
            '''
            if experiment_merge_strategy == MillipedeExperimentMergeStrategy.SUM:
                assert replicate_merge_strategy == MillipedeReplicateMergeStrategy.SUM, "replicate_merge_strategy must be SUM if experiment_merge_strategy is SUM"
            
            merged_experiment_df_list = []
            # Iterate through the experiments
            for experiment_index in range(len(enriched_pop_fn_experiment_list)):
                # Get the enriched_population and baseline_population for the experiment
                enriched_pop_fn = enriched_pop_fn_experiment_list[experiment_index]
                baseline_pop_fn = baseline_pop_fn_experiment_list[experiment_index]
            
            
                # Iterate through each replicate of the experiment
                merged_rep_df_list = []
                for rep in reps:
                    '''
                        Check file directories
                    '''
                    enriched_pop_full_fn_rep = (data_directory + '/' + enriched_pop_fn).format(rep)
                    baseline_pop_full_fn_rep = (data_directory + '/' + baseline_pop_fn).format(rep)
                    assert exists(enriched_pop_full_fn_rep), "File not found: {}".format(enriched_pop_full_fn_rep)
                    assert exists(baseline_pop_full_fn_rep), "File not found: {}".format(baseline_pop_full_fn_rep)

                    '''
                        Read in dataframes
                    '''
                    enriched_pop_rep_df = pd.read_csv(enriched_pop_full_fn_rep, sep='\t').fillna(value=0.0)
                    enriched_pop_nt_columns = [col for col in enriched_pop_rep_df.columns if ">" in col]
                    enriched_pop_rep_df = enriched_pop_rep_df[enriched_pop_nt_columns + [enriched_pop_df_reads_colname]]


                    baseline_pop_rep_df = pd.read_csv(baseline_pop_full_fn_rep, sep='\t').fillna(value=0.0)
                    baseline_pop_nt_columns = [col for col in baseline_pop_rep_df.columns if ">" in col]
                    baseline_pop_rep_df = baseline_pop_rep_df[baseline_pop_nt_columns + [baseline_pop_df_reads_colname]]

                    assert set(enriched_pop_nt_columns) == set(baseline_pop_nt_columns), "Nucleotide columns between enriched and baseline dataframes must be equivalent - are these screening the same regions?"
                    nucleotide_ids = enriched_pop_nt_columns

                    # Concat the enriched and baseline population dataframes together
                    merged_rep_df = pd.concat([enriched_pop_rep_df, baseline_pop_rep_df]).groupby(nucleotide_ids, as_index=False).sum()

                    # filter based on the per_replicate_each_condition_num_cutoff
                    merged_rep_df = merged_rep_df[merged_rep_df[baseline_pop_df_reads_colname] >= cutoff_specification.per_replicate_each_condition_num_cutoff]
                    merged_rep_df = merged_rep_df[merged_rep_df[enriched_pop_df_reads_colname] >= cutoff_specification.per_replicate_each_condition_num_cutoff]
                    merged_rep_df['total_reads'] = merged_rep_df[baseline_pop_df_reads_colname] + merged_rep_df[enriched_pop_df_reads_colname]

                    # filter on total reads based on the per_replicate_all_condition_num_cutoff
                    total_alleles_pre_filter = merged_rep_df.values.shape[0]
                    merged_rep_df = merged_rep_df[merged_rep_df["total_reads"] >= cutoff_specification.per_replicate_all_condition_num_cutoff]
                    
                    # Normalize counts
                    merged_rep_df = self.__normalize_counts(merged_rep_df, enriched_pop_df_reads_colname, baseline_pop_df_reads_colname, nucleotide_ids)
                    
                    # Add to the replicate list
                    merged_rep_df_list.append(merged_rep_df)
                    
                '''
                    Handle all replicates depending on provided strategy
                '''
                # If replicate_merge_strategy is SUM, sum the replicates together 
                if replicate_merge_strategy == MillipedeReplicateMergeStrategy.SUM:
                    nucleotide_ids = [col for col in merged_rep_df_list[0].columns if ">" in col]
                    merged_reps_df = pd.concat(merged_rep_df_list).groupby(nucleotide_ids, as_index=False).sum()
                    merged_reps_df = merged_reps_df[merged_reps_df["total_reads"] >= cutoff_specification.all_replicate_num_cutoff]
                    merged_experiment_df_list.append(merged_reps_df)
                elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.COVARIATE:
                    # DEVELOPER NOTE: Ensure that intercept_postfix between per-replicate and per-experiment are different
                    merged_reps_df = pd.concat([self.__get_intercept_df(merged_rep_df_list, experiment_id=experiment_index), pd.concat(merged_rep_df_list, ignore_index=True)], axis=1)
                    merged_experiment_df_list.append(merged_reps_df)
                elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.SEPARATE:
                    merged_experiment_df_list.append(merged_rep_df_list)
                elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.MODELLED_COMBINED:
                    # TODO: Perform error handling. Double check that each dataframe actually has a WT column
                    # This gets the WT allele from each replicate, as this will be used as the negative for CRISPR-Shrinkage
                    # Set negative counts
                    wt_allele_rep_df = [merged_rep_df[merged_rep_df[nucleotide_ids].sum(axis=1) == 0] for merged_rep_df in merged_rep_df_list]
                    
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
                    mut_allele_rep_df = [merged_rep_df[merged_rep_df[nucleotide_ids].sum(axis=1) > 0] for merged_rep_df in merged_rep_df_list]
                    
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

                    merged_reps_df = pd.concat([wt_allele_rep_df_merged_updated, mut_allele_rep_df_merged_updated], axis=0)
                    
                    merged_experiment_df_list.append(merged_reps_df)
                        
                # TODO: Any way to make more modular?
                elif replicate_merge_strategy == MillipedeReplicateMergeStrategy.MODELLED_SEPARATE:
                    # TODO: Perform error handling. Double check that each dataframe actually has a WT column
                    # This gets the WT allele from each replicate, as this will be used as the negative for CRISPR-Shrinkage
                    # Set negative counts
                    merged_rep_df_list_updated = []
                    for rep_i in reps:
                        merged_rep_df = merged_rep_df_list[rep_i]
                        wt_allele_df = merged_rep_df[merged_rep_df[nucleotide_ids].sum(axis=1) == 0]

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
                        mut_allele_df = merged_rep_df[merged_rep_df[nucleotide_ids].sum(axis=1) > 0]

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

                        merged_reps_df = pd.concat([wt_allele_rep_df_merged_updated, mut_allele_rep_df_merged_updated], axis=0)
                        
                        merged_rep_df_list_updated.append(merged_reps_df)
                    merged_experiment_df_list.append(merged_rep_df_list_updated)
                else:
                    raise Exception("Developer error: Unexpected value for MillipedeReplicateMergeStrategy: {}".format(replicate_merge_strategy))
            

            '''
                Handle all experiments depending on provided strategy
            '''
            __add_supporting_columns_partial = partial(self.__add_supporting_columns,
                                                       enriched_pop_df_reads_colname=enriched_pop_df_reads_colname,                               
                                                       baseline_pop_df_reads_colname= baseline_pop_df_reads_colname
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
                          nucleotide_ids: List[str]) -> pd.DataFrame:
        # TODO 5/15/23: Normalization is set to True always! Make it an input variable. Also, it should directly change the count rather than just the score
        # TODO 5/15/23: Also, allow normalization either by library size or by WT reads. For now, will just do WT reads
        wt_normalization = True
        
        
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
            
            encoding_df[enriched_pop_df_reads_colname] = enriched_read_counts
            encoding_df[baseline_pop_df_reads_colname] = baseline_read_counts
        
        return encoding_df

    def __add_supporting_columns(self, 
                                 encoding_df: pd.DataFrame, 
                                 enriched_pop_df_reads_colname: str, 
                                 baseline_pop_df_reads_colname: str
                                ) -> pd.DataFrame:
        # construct the simplest possible continuous-valued response variable.
        # this response variable is in [-1, 1]
        
        # Original
        enriched_read_counts = encoding_df[enriched_pop_df_reads_colname]
        baseline_read_counts = encoding_df[baseline_pop_df_reads_colname]
        total_read_counts = encoding_df['total_reads']
        
        if 'score' not in encoding_df.columns: 
            encoding_df['score'] = (enriched_read_counts - baseline_read_counts) / total_read_counts 
            
        # create scale_factor for normal likelihood model
        if 'scale_factor' not in encoding_df.columns: 
            encoding_df['scale_factor'] = 1.0 / np.sqrt(encoding_df['total_reads']) # NOTE: Intentionally keeping the total_reads as the raw to avoid being impact by normalization - this could be subject to change
        
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
    

millipede_model_specification_set : Mapping[str, MillipedeModelSpecification] = {
    "per_replicate_per_experiment_models" : MillipedeModelSpecification(
        model_types=[MillipedeModelType.NORMAL, MillipedeModelType.NORMAL_SIGMA_SCALED, MillipedeModelType.BINOMIAL],
        replicate_merge_strategy=MillipedeReplicateMergeStrategy.SEPARATE,
        experiment_merge_strategy=MillipedeExperimentMergeStrategy.SEPARATE,
        cutoff_specification=MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 2, 
            per_replicate_all_condition_num_cutoff = 0, 
            all_replicate_num_cutoff = 0, 
            all_experiment_num_cutoff = 0
        )
    ),
    "summed_replicate_per_experiment_models" : MillipedeModelSpecification(
        model_types=[MillipedeModelType.NORMAL, MillipedeModelType.NORMAL_SIGMA_SCALED, MillipedeModelType.BINOMIAL],
        replicate_merge_strategy=MillipedeReplicateMergeStrategy.SUM,
        experiment_merge_strategy=MillipedeExperimentMergeStrategy.SEPARATE,
        cutoff_specification=MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 0, 
            per_replicate_all_condition_num_cutoff = 0, 
            all_replicate_num_cutoff = 2, 
            all_experiment_num_cutoff = 0
        )
    ),
    "joint_replicate_per_experiment_models" : MillipedeModelSpecification(
        model_types=[MillipedeModelType.NORMAL, MillipedeModelType.NORMAL_SIGMA_SCALED, MillipedeModelType.BINOMIAL],
        replicate_merge_strategy=MillipedeReplicateMergeStrategy.COVARIATE,
        experiment_merge_strategy=MillipedeExperimentMergeStrategy.SEPARATE,
        cutoff_specification=MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 2, 
            per_replicate_all_condition_num_cutoff = 0, 
            all_replicate_num_cutoff = 0, 
            all_experiment_num_cutoff = 0
        )
    ),
    "joint_replicate_joint_experiment_models" : MillipedeModelSpecification(
        model_types=[MillipedeModelType.NORMAL, MillipedeModelType.NORMAL_SIGMA_SCALED, MillipedeModelType.BINOMIAL],
        replicate_merge_strategy=MillipedeReplicateMergeStrategy.COVARIATE,
        experiment_merge_strategy=MillipedeExperimentMergeStrategy.COVARIATE,
        cutoff_specification=MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 2, 
            per_replicate_all_condition_num_cutoff = 0, 
            all_replicate_num_cutoff = 0, 
            all_experiment_num_cutoff = 0
        )
    ),
    "summed_replicate_joint_experiment_models" : MillipedeModelSpecification(
        model_types=[MillipedeModelType.NORMAL, MillipedeModelType.NORMAL_SIGMA_SCALED, MillipedeModelType.BINOMIAL],
        replicate_merge_strategy=MillipedeReplicateMergeStrategy.SUM,
        experiment_merge_strategy=MillipedeExperimentMergeStrategy.COVARIATE,
        cutoff_specification=MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 0, 
            per_replicate_all_condition_num_cutoff = 2, 
            all_replicate_num_cutoff = 0, 
            all_experiment_num_cutoff = 0
        )
    ),
    "summed_replicate_summed_experiment_models" : MillipedeModelSpecification(
        model_types=[MillipedeModelType.NORMAL, MillipedeModelType.NORMAL_SIGMA_SCALED, MillipedeModelType.BINOMIAL],
        replicate_merge_strategy=MillipedeReplicateMergeStrategy.SUM,
        experiment_merge_strategy=MillipedeExperimentMergeStrategy.SUM,
        cutoff_specification=MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 0, 
            per_replicate_all_condition_num_cutoff = 0, 
            all_replicate_num_cutoff = 0, 
            all_experiment_num_cutoff = 2
        )
    )
}

# NOTE 5/15/23: Reduced specification set to the most important models
millipede_model_specification_set_jointonly : Mapping[str, MillipedeModelSpecification] = {
    "joint_replicate_per_experiment_models" : MillipedeModelSpecification(
        model_types=[MillipedeModelType.NORMAL, MillipedeModelType.NORMAL_SIGMA_SCALED, MillipedeModelType.BINOMIAL],
        replicate_merge_strategy=MillipedeReplicateMergeStrategy.COVARIATE,
        experiment_merge_strategy=MillipedeExperimentMergeStrategy.SEPARATE,
        cutoff_specification=MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 2, 
            per_replicate_all_condition_num_cutoff = 0, 
            all_replicate_num_cutoff = 0, 
            all_experiment_num_cutoff = 0
        )
    )
}

shrinkage_input_i = MillipedeShrinkageInput(
    include_observational_guides_in_fit = True,
    include_positive_control_guides_in_fit = True,
    sample_population_scaling_factors_exp_list = np.asarray([[500, 500, 500], [500, 500, 500]]),
    control_population_scaling_factors_exp_list = np.asarray([[500, 500, 500], [500, 500, 500]]),
    deviation_weights = np.asarray([1,1,1]),
    KL_guide_set_weights = None,
    shrinkage_prior_strength =  None, 
    posterior_estimator = "mean",
    random_seed = 234,
    cores=100)

millipede_model_specification_set_shrinkage : Mapping[str, MillipedeModelSpecification] = {
    "modelled_combined_replicate_per_experiment_models" : MillipedeModelSpecification(
        model_types=[MillipedeModelType.NORMAL_SIGMA_SCALED],
        replicate_merge_strategy=MillipedeReplicateMergeStrategy.MODELLED_COMBINED,
        experiment_merge_strategy=MillipedeExperimentMergeStrategy.SEPARATE,
        cutoff_specification=MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 2, 
            per_replicate_all_condition_num_cutoff = 0, 
            all_replicate_num_cutoff = 0, 
            all_experiment_num_cutoff = 0
        ),
        shrinkage_input = shrinkage_input_i
    )
    #, TODO (3/16/2023): Fix bug in separate run, but for now priority is on combined.  
    #"summed_replicate_per_experiment_models" : MillipedeModelSpecification(
    #    model_types=[MillipedeModelType.NORMAL_SIGMA_SCALED],
    #    replicate_merge_strategy=MillipedeReplicateMergeStrategy.MODELLED_COMBINED,
    #    experiment_merge_strategy=MillipedeExperimentMergeStrategy.SEPARATE,
    #    cutoff_specification=MillipedeCutoffSpecification(
    #        per_replicate_each_condition_num_cutoff = 0, 
    #        per_replicate_all_condition_num_cutoff = 0, 
    #        all_replicate_num_cutoff = 2, 
    #        all_experiment_num_cutoff = 0
    #    ),
    #    shrinkage_input = shrinkage_input_i
    #)
}










# TODO: Make into pydantic or dataclass
@dataclass
class MillipedeModelSpecificationSingleMatrixResult:
    millipede_model_specification_single_matrix_result: Mapping[MillipedeModelType, Union[NormalLikelihoodVariableSelector, BinomialLikelihoodVariableSelector]]
        
@dataclass
class MillipedeModelSpecificationResult:
    millipede_model_specification_result_input: Union[MillipedeModelSpecificationSingleMatrixResult, List[MillipedeModelSpecificationSingleMatrixResult], List[List[MillipedeModelSpecificationSingleMatrixResult]]]
    millipede_model_specification: MillipedeModelSpecification
    millipede_input_data: MillipedeInputData


class MillipedeComputeDevice(Enum):
    CPU: str = "cpu"
    GPU: str = "gpu"

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
                                                                   S=1, 
                                                                   tau=0.01,
                                                                   tau_intercept=1.0e-4,
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
                                                                                S=1, 
                                                                                tau=0.01,
                                                                                tau_intercept=1.0e-4,
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
                                                                       S=1, 
                                                                       tau=0.01,
                                                                       tau_intercept=1.0e-4,
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

parse_position = lambda feature: int(feature[:feature.index(">")-1])
parse_ref = lambda feature: feature[feature.index(">")-1:feature.index(">")]
parse_alt = lambda feature: feature[feature.index(">")+1:]


def plot_millipede_scores(score_df, encoding, original_seq):
    # Set heatmap meta
    
    columns = [feature for feature in encoding.columns if ">" in feature ]
    feature_positions = [parse_position(feature) if ">" in feature else None for feature in columns]
    feature_refs = [parse_ref(feature) if ">" in feature else None for feature in columns]
    feature_alts = [parse_alt(feature) if ">" in feature else None for feature in columns]
    feature_meta_df = pd.DataFrame({"positions": feature_positions, "refs": feature_refs, "alts": feature_alts})
    feature_meta_df.index = columns
    
    
    xlabels = set(feature_positions)
    ylabels = ["A", "C", "T", "G"]
    position_size = len(xlabels)
    nt_size = len(ylabels)
    x, y = np.meshgrid(np.arange(position_size), np.arange(nt_size))

    # Get heatmap inputs
    coef_alt_list = []
    pip_alt_list = []
    for alt_index, alt in enumerate(ylabels):
        coef_alt_position_array = np.asarray([])
        pip_alt_position_array = np.asarray([])
        for pos_index, pos in enumerate(xlabels):
            ref = original_seq[pos]
            if alt == ref:
                coef_alt_position_array = np.append(coef_alt_position_array, 0)
                pip_alt_position_array = np.append(pip_alt_position_array, 0)
            else:
                feature_name = str(pos) + ref + ">" + alt
                coef_score = score_df.loc[feature_name, "Coefficient"]
                pip_score = score_df.loc[feature_name, "PIP"]
                coef_alt_position_array = np.append(coef_alt_position_array, coef_score)
                pip_alt_position_array = np.append(pip_alt_position_array, pip_score)
        coef_alt_list.append(coef_alt_position_array)
        pip_alt_list.append(pip_alt_position_array)

    coef_df = np.vstack(coef_alt_list)
    pip_df = np.vstack(pip_alt_list)




    # Process heatmap inputs
    pip_upper_threshold = 0.3
    pip_df[np.where(pip_df > pip_upper_threshold)] = pip_upper_threshold
    ref_indices = [ylabels.index(original_seq[position]) for position in xlabels]



    fig, ax = plt.subplots(figsize=(100,4))

    R = pip_df/pip_upper_threshold/2
    circles = [plt.Circle((j,i), radius=r, edgecolor="black", linewidth=15) for r, j, i in zip(R.flat, x.flat, y.flat)]
    col = PatchCollection(circles, array=coef_df.flatten(), cmap="RdBu")
    col.set_clim([-1, 1])

    ax.add_collection(col)
    ax.set(xticks=np.arange(position_size), yticks=np.arange(nt_size),
           xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(position_size+1)-0.5, minor=True)
    ax.set_yticks(np.arange(nt_size+1)-0.5, minor=True)
    ax.grid(which='minor')
    ax.scatter(list(xlabels), ref_indices)

    ax.set_xlim(-0.5, position_size+0.5)

    fig.colorbar(col)
    plt.xticks(rotation = 45)
    plt.show()

from typing import Union, List
from os.path import exists

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

# Draft new heatmap with differnet implementation 8/30/2022
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing_extensions import TypeGuard
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

# It seems the best way is an interactive web app for this data - start thinking about how this can be coded in plotly.
class MillipedeModelResultVisualizationExperimentalGroup:
    
    def __init__(self, millipede_model_experimental_group: MillipedeModelExperimentalGroup, 
                 raw_encoding_dfs_experimental_group: RawEncodingDataframesExperimentalGroup, 
                 raw_encodings_editing_freqs_experimental_group: EncodingEditingFrequenciesExperimentalGroup,
                 enriched_pop_label: str,
                 baseline_pop_label: str,
                 presort_pop_label: Optional[str]=None,
                 ctrl_pop_label: Optional[str]=None):
        self.millipede_model_experimental_group = millipede_model_experimental_group
        self.raw_encoding_dfs_experimental_group = raw_encoding_dfs_experimental_group
        self.raw_encodings_editing_freqs_experimental_group = raw_encodings_editing_freqs_experimental_group
        self.enriched_pop_label = enriched_pop_label
        self.baseline_pop_label = baseline_pop_label
        self.presort_pop_label = presort_pop_label
        self.ctrl_pop_label = ctrl_pop_label
        
        #plot_millipede_model_wrapper(millipede_model_experimental_group=millipede_model_experimental_group, 
        #                             raw_encoding_dfs_experimental_group=raw_encoding_dfs_experimental_group, 
        #                             raw_encodings_editing_freqs_experimental_group=raw_encodings_editing_freqs_experimental_group,
        #                             enriched_pop_label=enriched_pop_label,
        #                             baseline_pop_label=baseline_pop_label,
        #                             presort_pop_label=presort_pop_label,
        #                             ctrl_pop_label=ctrl_pop_label)
    
    
    def plot_all_for_model_specification_id(self, 
                                            reference_seq: str,
                                            model_specification_id_list: Optional[List[str]]=None,
                                            model_specification_label_list: Optional[List[str]]=None,
                                            model_types: Optional[List[MillipedeModelType]] = None,
                                            experiment_indices: Optional[List[int]]=None,
                                            replicate_indices: Optional[List[int]]=None,
                                            pdf_filename: Optional[str]= None):
       
        pdf: Optional[PdfPages] = None
        if pdf_filename != None:
            pdf = PdfPages(pdf_filename)
            
        if model_specification_id_list == None:
            model_specification_id_list = self.millipede_model_experimental_group.millipede_model_specification_set_with_results.keys()
        if model_specification_label_list == None:
            model_specification_label_list = model_specification_id_list
          
        assert len(model_specification_label_list) == len(model_specification_id_list), "Model specification label list (len={}) and ID list (len={}) must be same length".format(len(model_specification_label_list), len(model_specification_id_list))
           
        experiment_labels = self.millipede_model_experimental_group.experiments_inputdata.experiment_labels
        reps_labels = self.millipede_model_experimental_group.experiments_inputdata.reps
        for model_specification_id_index, model_specification_id in enumerate(model_specification_id_list):
            model_specification_label = model_specification_label_list[model_specification_id_index]
            
            # Retrieve result(s) for specified specification
             
            millipede_model_specification_result_wrapper: MillipedeModelSpecificationResult= self.millipede_model_experimental_group.millipede_model_specification_set_with_results[model_specification_id]
            millipede_model_specification: MillipedeModelSpecification= millipede_model_specification_result_wrapper.millipede_model_specification
            millipede_model_specification_result_object: Union[MillipedeModelSpecificationSingleMatrixResult, List[MillipedeModelSpecificationSingleMatrixResult], List[List[MillipedeModelSpecificationSingleMatrixResult]]] = millipede_model_specification_result_wrapper.millipede_model_specification_result_input 
            
            # NOTE 03202023: If only a single matrix is provided (i.e. joint model)
            if isinstance(millipede_model_specification_result_object, MillipedeModelSpecificationSingleMatrixResult):
                if experiment_indices != None:
                    print("experiment_indices provided but will not be used as replicates seemed to have been merged")
                if replicate_indices != None:
                    print("replicate_indices provided but will not be used as replicates seemed to have been merged")


                if model_types != None:
                    for model_type in model_types:
                        assert model_type in millipede_model_specification_result_object.millipede_model_specification_single_matrix_result.keys(), "No model for provided model type: {}".format(model_type)
                elif model_types == None:
                    model_types = list(millipede_model_specification_result_object.millipede_model_specification_single_matrix_result.keys())

                for model_type in model_types:
                    print("Showing results for model_type: {}".format(model_type))

                    millipede_model_score_df = millipede_model_specification_result_object.millipede_model_specification_single_matrix_result[model_type].summary.sort_values(by=['PIP'], ascending=False)
                    baseline_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg[0]
                    enriched_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.enriched_pop_encoding_editing_freq_avg[0]
                    presort_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.presort_pop_encoding_editing_freq_avg[0]
                    ctrl_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.ctrl_pop_encoding_editing_freq_avg[0]

                    experiment_label = "+".join(experiment_labels) + " Merge Strategy={}".format(millipede_model_specification.experiment_merge_strategy.value)
                    replicate_label = "+".join(map(str, reps_labels)) + " Merge Strategy={}".format(millipede_model_specification.replicate_merge_strategy.value)
                    model_type_label=model_type.value
                    base_title = "Specification={}; Experiment={}; Replicate={}; Model={}".format(model_specification_label, experiment_label, replicate_label, model_type_label)
                    self.__plot_millipede_model_wrapper(millipede_model_score_df=millipede_model_score_df, 
                                                        original_seq=reference_seq,
                                                        baseline_pop_editing_frequency_avg=baseline_pop_editing_frequency_avg,
                                                        enriched_pop_editing_frequency_avg=enriched_pop_editing_frequency_avg,
                                                        presort_pop_editing_frequency_avg=presort_pop_editing_frequency_avg,
                                                        ctrl_pop_editing_frequency_avg=ctrl_pop_editing_frequency_avg,
                                                        enriched_pop_label=self.enriched_pop_label,
                                                        baseline_pop_label=self.baseline_pop_label,
                                                        presort_pop_label=self.presort_pop_label,
                                                        ctrl_pop_label=self.ctrl_pop_label,
                                                        base_title=base_title,
                                                        pdf=pdf)
            # NOTE 03202023: If only a list of matrices is provided
            elif MillipedeModelResultVisualizationExperimentalGroup.__is_list_of_millipede_model_specification_single_matrix_result(millipede_model_specification_result_object): # -> TypeGuard[List[MillipedeModelSpecificationSingleMatrixResult]]
                millipede_model_specification_result_object: List[MillipedeModelSpecificationSingleMatrixResult]
                if experiment_indices != None:
                    for experimental_index in experiment_indices:
                        assert experimental_index in range(len(millipede_model_specification_result_object)), "Provided experiment_index {} out of range".format(experimental_index) 
                else:
                    experiment_indices = range(len(millipede_model_specification_result_object))

                if replicate_indices != None:
                    print("replicate_index provided but will not be used as replicates seemed to have been merged")

                # Iterate through the experiments in the results object
                for experiment_index in experiment_indices:
                    print("Showing results for experiment index: {}".format(experiment_index))
                    millipede_model_specification_result_object_exp = millipede_model_specification_result_object[experiment_index]

                    # Subset the model types 
                    if model_types != None:
                        for model_type in model_types:
                            # NOTE: Bug in assert statement by comparing object hashes. Instead, compare the enum values
                            # assert model_type in millipede_model_specification_result_object_exp.millipede_model_specification_single_matrix_result.keys(), "No model for provided model type: {}".format(model_type)
                            # TODO: Include the code that actually subsets the model types
                            pass
                    elif model_types == None:
                        model_types = list(millipede_model_specification_result_object_exp.millipede_model_specification_single_matrix_result.keys())



                    for model_type_i, model_type in enumerate(model_types):
                        print("Showing results for model_type {}: {}".format(model_type_i, model_type))
                        millipede_model_score_df = millipede_model_specification_result_object_exp.millipede_model_specification_single_matrix_result[model_type].summary.sort_values(by=['PIP'], ascending=False)
                        print("HERE1")
                        print(len(self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg))
                        baseline_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg[1][experiment_index]
                        enriched_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.enriched_pop_encoding_editing_freq_avg[1][experiment_index]
                        presort_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.presort_pop_encoding_editing_freq_avg[1][experiment_index]
                        ctrl_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.ctrl_pop_encoding_editing_freq_avg[0] # TODO: Hardcoding to the first provided control, but I guess there could be more?

                        experiment_label = experiment_labels[experiment_index]
                        replicate_label = "+".join(map(str, reps_labels)) + " Merge Strategy={}".format(millipede_model_specification.replicate_merge_strategy.value)
                        model_type_label=model_type.value
                        base_title = "Specification={}; Experiment={}; Replicate={}; Model={}".format(model_specification_label, experiment_label, replicate_label, model_type_label)
                        self.__plot_millipede_model_wrapper(millipede_model_score_df=millipede_model_score_df,
                                                            original_seq=reference_seq, 
                                                            baseline_pop_editing_frequency_avg=baseline_pop_editing_frequency_avg,
                                                            enriched_pop_editing_frequency_avg=enriched_pop_editing_frequency_avg,
                                                            presort_pop_editing_frequency_avg=presort_pop_editing_frequency_avg,
                                                            ctrl_pop_editing_frequency_avg=ctrl_pop_editing_frequency_avg,
                                                            enriched_pop_label=self.enriched_pop_label,
                                                            baseline_pop_label=self.baseline_pop_label,
                                                            presort_pop_label=self.presort_pop_label,
                                                            ctrl_pop_label=self.ctrl_pop_label,
                                                            base_title=base_title,
                                                            pdf=pdf)


            elif MillipedeModelResultVisualizationExperimentalGroup.__is_list_of_list_millipede_model_specification_single_matrix_result(millipede_model_specification_result_object): # -> TypeGuard[List[List[MillipedeModelSpecificationSingleMatrixResult]]]
                millipede_model_specification_result_object: List[List[MillipedeModelSpecificationSingleMatrixResult]]

                if experiment_indices != None:
                    for experimental_index in experiment_indices:
                        assert experimental_index in range(len(millipede_model_specification_result_object)), "Provided experiment_index {} out of range".format(experiment_index) 

                else:
                    experiment_indices = range(len(millipede_model_specification_result_object))

                if replicate_indices != None:
                    for millipede_model_specification_result_rep_object in millipede_model_specification_result_object:
                        for replicate_index in replicate_indices:
                            assert replicate_index in range(len(millipede_model_specification_result_rep_object)), "Provided experiment_index {} out of range".format(replicate_index)

                for millipede_model_specification_result_rep_object in millipede_model_specification_result_object:
                    if replicate_indices == None:
                        replicate_indices =  range(len(millipede_model_specification_result_rep_object))

                for experiment_index in experiment_indices:
                    print("Showing results for experiment index: {}".format(experiment_index))
                    millipede_model_specification_result_object_exp = millipede_model_specification_result_object[experiment_index]

                    for replicate_index in replicate_indices:
                        print("Showing results for replicate index: {}".format(replicate_index))
                        millipede_model_specification_result_object_exp_rep = millipede_model_specification_result_object_exp[replicate_index]

                                 # Execute
                        if model_types != None:
                            for model_type in model_types:
                                assert model_type in millipede_model_specification_result_object_exp_rep.millipede_model_specification_single_matrix_result.keys(), "No model for provided model type: {}".format(model_type)
                        elif model_types == None:
                            model_types = list(millipede_model_specification_result_object_exp_rep.millipede_model_specification_single_matrix_result.keys())

                        for model_type in model_types:
                            print("Showing results for model_type: {}".format(model_type))

                            millipede_model_score_df = millipede_model_specification_result_object_exp_rep.millipede_model_specification_single_matrix_result[model_type].summary.sort_values(by=['PIP'], ascending=False)
                            print("HERE2")
                            print(len(self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg))
                            baseline_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg[1][experiment_index][replicate_index] # NOTE 3/20/2023: Unsure if retrieving replice frequencies appropriately.
                            enriched_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.enriched_pop_encoding_editing_freq_avg[1][experiment_index][replicate_index] # NOTE 3/20/2023: Unsure if retrieving replice frequencies appropriately.
                            presort_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.presort_pop_encoding_editing_freq_avg[1][experiment_index][replicate_index] # NOTE 3/20/2023: Unsure if retrieving replice frequencies appropriately.
                            ctrl_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.ctrl_pop_encoding_editing_freq_avg[0][replicate_index] # TODO: Hardcoding to the first provided control, but I guess there could be more?

                            experiment_label = experiment_labels[experiment_index]
                            replicate_label = reps_labels[replicate_index]
                            model_type_label=model_type.value
                            base_title = "Specification={}; Experiment={}; Replicate={}; Model={}".format(model_specification_label, experiment_label, replicate_label, model_type_label)
                            self.__plot_millipede_model_wrapper(millipede_model_score_df=millipede_model_score_df, 
                                                                baseline_pop_editing_frequency_avg=baseline_pop_editing_frequency_avg,
                                                                enriched_pop_editing_frequency_avg=enriched_pop_editing_frequency_avg,
                                                                presort_pop_editing_frequency_avg=presort_pop_editing_frequency_avg,
                                                                ctrl_pop_editing_frequency_avg=ctrl_pop_editing_frequency_avg,
                                                                enriched_pop_label=self.enriched_pop_label,
                                                                baseline_pop_label=self.baseline_pop_label,
                                                                presort_pop_label=self.presort_pop_label,
                                                                ctrl_pop_label=self.ctrl_pop_label,
                                                                base_title=base_title,
                                                                pdf=pdf)

            else:
                raise Exception("Unexpected type for millipede_model_specification_result_rep_object")

        if pdf != None:
            pdf.close()
           
    @staticmethod
    def __is_list_of_millipede_model_specification_single_matrix_result(val: List[object]) -> TypeGuard[List[MillipedeModelSpecificationSingleMatrixResult]]:
        return all(isinstance(x, MillipedeModelSpecificationSingleMatrixResult) for x in val)
    
    @staticmethod
    def __is_list_of_list_millipede_model_specification_single_matrix_result(val: List[object]) -> TypeGuard[List[List[MillipedeModelSpecificationSingleMatrixResult]]]:
        if all(isinstance(x, list) for x in val) != True:
            return False
        return all(isinstance(y, MillipedeModelSpecificationSingleMatrixResult) for x in val for y in x)

    '''
        Define feature dataframe
    '''
    def __generate_millipede_heatmap_input(self, millipede_model_score_df: pd.DataFrame, original_seq: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nucleotide_ids = [feature for feature in millipede_model_score_df.index if ">" in feature]
        feature_positions = [parse_position(feature) if ">" in feature else None for feature in nucleotide_ids]
        feature_refs = [parse_ref(feature) if ">" in feature else None for feature in nucleotide_ids]
        feature_alts = [parse_alt(feature) if ">" in feature else None for feature in nucleotide_ids]
        feature_meta_df = pd.DataFrame({"positions": feature_positions, "refs": feature_refs, "alts": feature_alts})
        feature_meta_df.index = nucleotide_ids
        feature_meta_df = feature_meta_df.sort_values(["positions", "alts"])
        feature_meta_positions = feature_meta_df[["positions", "refs"]].drop_duplicates()
        
        # Define x-axis and y-axis labels
        xlabels_POS = [format(pos, '03d') + "_" + feature_meta_positions["refs"][i] for i, pos in enumerate(feature_meta_positions["positions"])]
        ylabels_ALT = ["A", "C", "T", "G"]
        
        position_size = len(xlabels_POS)
        nt_size = len(ylabels_ALT)

        # Get heatmap-specific inutsinputs
        coef_position_list = []
        pip_position_list = []
        xlabel_position_list = []
        ylabel_position_list = []

        for position_index, position in enumerate(set(feature_meta_df["positions"])):
            coef_alt_position_array = np.asarray([]) # Model feature coefficient per position and per alt array
            pip_alt_position_array = np.asarray([]) # Model feature PIP per position and per alt array
            xlabel_position_array = np.asarray([]) # Heatmap x-index for each feature
            ylabel_position_array = np.asarray([]) # Heatmap y-index for each feature

            # Get the ref and alt feature for each position
            ref_nt: str = original_seq[position]
            for alt_index, alt_nt in enumerate(ylabels_ALT):
                xlabel_position_array = np.append(xlabel_position_array, xlabels_POS[position]) # Add the X-label, which is from xlabels_POS
                ylabel_position_array = np.append(ylabel_position_array, alt_nt) # Add the y-label, which is the alt_nt


                # If alt and ref are the same, there is no feature for this in the model, so make 0
                if alt_nt == ref_nt:
                    coef_alt_position_array = np.append(coef_alt_position_array, 0)
                    pip_alt_position_array = np.append(pip_alt_position_array, 0)
                else:
                    feature_name = str(position) + ref_nt + ">" + alt_nt  
                    coef_score = millipede_model_score_df.loc[feature_name, "Coefficient"]
                    pip_score = millipede_model_score_df.loc[feature_name, "PIP"]
                    coef_alt_position_array = np.append(coef_alt_position_array, coef_score)
                    pip_alt_position_array = np.append(pip_alt_position_array, pip_score)

            coef_position_list.append(coef_alt_position_array)
            pip_position_list.append(pip_alt_position_array)
            xlabel_position_list.append(xlabel_position_array)
            ylabel_position_list.append(ylabel_position_array)


        coef_vstack = np.vstack(coef_position_list)
        pip_vstack = np.vstack(pip_position_list)
        xlabel_vstack = np.vstack(xlabel_position_list)
        ylabel_vstack = np.vstack(ylabel_position_list)

        coef_flatten = coef_vstack.flatten()
        pip_flatten = pip_vstack.flatten()
        xlabel_flatten = xlabel_vstack.flatten()
        ylabel_flatten = ylabel_vstack.flatten()

        return (coef_flatten, pip_flatten, xlabel_flatten, ylabel_flatten)
   
    
    # Step 1 - Make a scatter plot with square markers, set column names as labels
    def __millipede_results_heatmap(self, x: pd.Series, 
                                    y: pd.Series, 
                                    size: pd.Series, 
                                    color:pd.Series,
                                    alpha: pd.Series, 
                                    editing_freq_groups: List[pd.Series], 
                                    editing_freq_groups_labels: List[str], 
                                    baseline_group_index: int, 
                                    selection_group_index: int, 
                                    overall_group_index: int, 
                                    base_title: Optional[str]=None,
                                    frequency_max: Optional[float]=None, 
                                    frequency_min: Optional[float]=None,
                                    pdf: Optional[PdfPages] = None):
        '''
            3 axes
            axes[0] is the bar plot of per-position mutational frequency
            axes[1] is the dot plot of per-position enrichment
            axes[2] is the heatmap of the Millipede model coefficients 
        '''
        #scale = 0.5# 0.38
        scale=0.25
        #fig, axes = plt.subplots(nrows=3, ncols= 1, figsize=(115*scale,20*scale))
        fig, axes = plt.subplots(nrows=3, ncols= 1, figsize=(30*scale,30*scale)) # TODO ADDED MANUAY 
        axes[0].tick_params(axis='x', which='major', labelsize=8)
        axes[1].tick_params(axis='x', which='major', labelsize=8)
        axes[2].tick_params(axis='x', which='major', labelsize=8)
        fig.suptitle(base_title, fontsize=16)
        fig.tight_layout(pad=2)

        '''
            Preprocess inputs
        '''
        # Mapping from column names to integer coordinates
        x_labels = [v for v in sorted(x.unique())]
        y_labels = [v for v in sorted(y.unique())]
        x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
        y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
        alpha[alpha>1] = 1

        '''
            Configure axes[0]
        '''
        width = 1./(len(editing_freq_groups)+1)
        for index, editing_freq_group in enumerate(editing_freq_groups):
            bar_xval_start = - (width * ((len(editing_freq_groups)+1)/2.))  
            bar_xval_offset = width*(index+1) 
            #print(bar_xval_start)
            axes[0].bar(editing_freq_group.index.values + bar_xval_start + bar_xval_offset, editing_freq_group.values, width, label=editing_freq_groups_labels[index])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axes[0].set_ylabel('Editing Frequency')
        #axes[0].set_xticks([x_to_num[v] for v in x_labels], weight ='bold', labels=[x_to_num[v] for v in x_labels], rotation=45, horizontalalignment='right') TODO MANUALLY REMOVED
        #axes[0].set_xticklabels()
        #axes[1].set_xticklabels(range(len(xlabels)), rotation=45, horizontalalignment='right')
        axes[0].set_xlim(0-1, len(x_labels)+1)
        #axes[0].set_ylim(0, 0.05) # ADDED MANUALLY TODO
        axes[0].set_xlim(200, 250) # ADDED MANUALLY TODO
        #axes[0].grid(which='major') # TODO REMOVED MANUALLY
        axes[0].legend(loc=2)


        '''
            Configure axes[1]
        '''
        enrichment_scores = editing_freq_groups[selection_group_index]/(editing_freq_groups[selection_group_index] + editing_freq_groups[baseline_group_index])
        frequency_scores = editing_freq_groups[overall_group_index]
        cmap = plt.get_cmap("inferno")

        frequency_max_input = np.max(frequency_scores) if frequency_max == None else frequency_max
        frequency_min_input = np.min(frequency_scores) if frequency_min == None else frequency_min

        rescale = lambda y: (y - frequency_min_input) / (frequency_max_input - frequency_min_input)
        rects = axes[1].scatter(range(len(enrichment_scores)), enrichment_scores, color = cmap(rescale(frequency_scores)), s=30)
        axes[1].set_xticks([x_to_num[v] for v in x_labels], weight ='bold', labels=[x_to_num[v] for v in x_labels], rotation=45, horizontalalignment='right')
        #axes[1].set_xticklabels()
        axes[1].set_ylim(-0.1,1.1)
        axes[1].set_xlim(0-1, len(x_labels)+1)
        axes[1].set_xlim(200, 250) # ADDED MANUALLY TODO
        axes[1].axhline(y=0.5, color='black', linestyle='dotted', linewidth=1)
        axes[1].grid(which='major')
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(frequency_min_input,frequency_max_input))
        sm.set_array([])

        cbaxes = inset_axes(axes[1], width="5%", height="5%", loc=2) 
        cbar = plt.colorbar(sm, cax=cbaxes, orientation='horizontal')
        #cbar.set_label('Corrected Mutational Frequency')

        '''
            Configure axes[2]
        '''
        cmap = LinearSegmentedColormap.from_list('tricolor', ['#FF0000', '#FFFFFF', '#0000FF'])

        # Set the range for the color mapping
        vmin = -0.5
        vmax = 0.5

        # Map the values in the z array to colors using the colormap
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        color_values = cmap(norm(color))


        size_scale = 200
        axes[2].scatter(
            x=x.map(x_to_num), # Use mapping for x
            y=y.map(y_to_num), # Use mapping for y
            s=alpha * size_scale, # Vector of square sizes, proportional to size parameter
            color= color_values,
            marker='s' # Use square as scatterplot marker
        )

        # Show column labels on the axes
        axes[2].set_xticks([x_to_num[v] for v in x_labels], weight = 'bold', labels=x_labels, rotation=45, horizontalalignment='right')
        #axes[2].set_xticklabels()
        axes[2].set_xlim(0-1, len(x_labels)+1)
        axes[2].set_xlim(200, 250) # ADDED MANUALLY TODO
        axes[2].set_yticks([y_to_num[v] for v in y_labels])
        axes[2].set_yticklabels(y_labels)
        axes[2].grid(which='major')
        
        if False:# TODO: Temporary removal
            sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin,vmax))
            sm.set_array([])

            cbaxes = inset_axes(axes[2], width="5%", height="5%", loc=2) 
            cbar = plt.colorbar(sm, cax=cbaxes, orientation='horizontal')
            cbar.set_label('Coefficient')
        
        labels = np.asarray([0.1, 0.5, 1.0])
        markersizes = labels*size_scale
        for i in range(len(markersizes)):
            axes[2].scatter([], [], s=markersizes[i], label=labels[i], marker='s', color="blue")

        # Add legend based on the dummy scatter plot
        axes[2].legend(title='PIP', labelspacing=0.7, borderpad=0.3, loc='lower left')


        pdf.savefig(fig)
        plt.show()
    
    
    def __plot_millipede_model_wrapper(self, millipede_model_score_df: pd.DataFrame, 
                                       original_seq:str,
                                       baseline_pop_editing_frequency_avg: pd.Series,
                                       enriched_pop_editing_frequency_avg: pd.Series,
                                       enriched_pop_label: str,
                                       baseline_pop_label: str,
                                       presort_pop_label: Optional[str]=None,
                                       ctrl_pop_label: Optional[str]=None,
                                       base_title: Optional[str]=None,
                                       presort_pop_editing_frequency_avg: Optional[pd.Series] = None,
                                       ctrl_pop_editing_frequency_avg: Optional[pd.Series] = None,
                                       pdf: Optional[PdfPages] = None):
        
        coef_flatten, pip_flatten, xlabel_flatten, ylabel_flatten = self.__generate_millipede_heatmap_input(millipede_model_score_df, original_seq)
        
        print([presort_pop_editing_frequency_avg, enriched_pop_editing_frequency_avg, baseline_pop_editing_frequency_avg, ctrl_pop_editing_frequency_avg])
        self.__millipede_results_heatmap(
            x=pd.Series(xlabel_flatten),
            y=pd.Series(ylabel_flatten),
            size=pd.Series(coef_flatten).abs(),
            color=pd.Series(coef_flatten),
            alpha=pd.Series(pip_flatten).abs(),
            base_title=base_title,
            editing_freq_groups = [presort_pop_editing_frequency_avg, enriched_pop_editing_frequency_avg, baseline_pop_editing_frequency_avg, ctrl_pop_editing_frequency_avg],
            editing_freq_groups_labels = [presort_pop_label, enriched_pop_label, baseline_pop_label, ctrl_pop_label],
            baseline_group_index = 2,
            selection_group_index = 1,
            overall_group_index = 0,
            frequency_min = 0,
            pdf = pdf
        )

        self.__millipede_results_heatmap(
            x=pd.Series(xlabel_flatten),
            y=pd.Series(ylabel_flatten),
            size=pd.Series(coef_flatten).abs(),
            color=pd.Series(coef_flatten),
            alpha=pd.Series(pip_flatten).abs(),
            base_title=base_title,
            editing_freq_groups = [presort_pop_editing_frequency_avg-ctrl_pop_editing_frequency_avg, enriched_pop_editing_frequency_avg-ctrl_pop_editing_frequency_avg, baseline_pop_editing_frequency_avg-ctrl_pop_editing_frequency_avg, ctrl_pop_editing_frequency_avg-ctrl_pop_editing_frequency_avg],
            #editing_freq_groups_labels = [presort_pop_label + "-" + ctrl_pop_label, enriched_pop_label + "-" + ctrl_pop_label, baseline_pop_label + "-" + ctrl_pop_label, ctrl_pop_label + "-" + ctrl_pop_label],
            editing_freq_groups_labels = [presort_pop_label, enriched_pop_label, baseline_pop_label, ctrl_pop_label], # TODO: TEMPORARY for presentation
            baseline_group_index = 2,
            selection_group_index = 1,
            overall_group_index = 0,
            frequency_min = 0,
            pdf = pdf
        )
    
    
#millipede_model_experimental_group: MillipedeModelExperimentalGroup, 
#raw_encoding_dfs_experimental_group: RawEncodingDataframesExperimentalGroup, 
#raw_encodings_editing_freqs_experimental_group: EncodingEditingFrequenciesExperimentalGroup, 
#reference_sequence: str
         