from .models_processing import *
from .models_inputs import *

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