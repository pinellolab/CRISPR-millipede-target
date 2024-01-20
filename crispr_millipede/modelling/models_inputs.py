import numpy as np
from dataclasses import dataclass
from typing import Union, Mapping, List, Tuple
from enum import Enum
import pandas as pd
from millipede import NormalLikelihoodVariableSelector
from millipede import BinomialLikelihoodVariableSelector


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