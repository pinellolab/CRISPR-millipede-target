# CRISPR-Millipede

The CRISPR-Millipede Python package is used to <ins> process targeted amplicon-sequencing of tiled sequences from base-editing tiling screens to identify functional nucleotides</ins>. 

## Installation
CRISPR-Millipede can be installed from PyPi `pip install crispr-millipede`. CRISPR-Millipede requires Python versions >=3.8,<3.12.

## Perform encoding of targeted amplicon-sequencing data
CRISPR-Millipede takes in as input the allele frequency tables produced from [CRISPResso2](https://github.com/pinellolab/CRISPResso2). Refer to the [CRISPResso2 documentation](https://github.com/pinellolab/CRISPResso2) for instructions on how to run CRISPResso2. An example command for CRISPResso2 in batch mode is as follows:

```
CRISPRessoBatch -bs {FASTQ_FILENAME} -a {AMPLICON_SEQUENCE} -an cd19 -q 30 --exclude_bp_from_left 3 --exclude_bp_from_right 3 --no_rerun -n {SCREEN_NAME} --min_frequency_alleles_around_cut_to_plot 0.001 --max_rows_alleles_around_cut_to_plot 500 -p 20  --plot_window_size 4 --base_editor_output -w 0 -bo {OUTPUT_DIRECTORY}
```

For each sample, CRISPResso2 will produce a allele frequency table named "Alleles_frequency_table.zip" which is used as input to the CRISPR-Millipede package.

Import and prepare the parameters of the encoding step by passing in the amplicon_sequence (required), the acceptable variant types (optional), population colummn suffixes for indexing (required), and encoding edge trimming for reducing sequencing background (optional) to the `EncodingParameters` class.

```
from crispr_millipede import encoding as cme

AMPLICON = "ACTGACTG...."
ABE_VARIANT_TYPES = [("A", "G"), ("T", "C")]
CBE_VARIANT_TYPES = [("C", "T"), ("G", "A")]

encoding_parameters = cme.EncodingParameters(complete_amplicon_sequence=AMPLICON,
                            population_baseline_suffix="_BASELINE",
                            population_target_suffix="_TARGET",
                            population_presort_suffix="_PRESORT",
                            wt_suffix="_WT",
                            trim_left=20,
                            trim_right=20,
                            variant_types=ABE_VARIANT_TYPES,
                            remove_denoised=True)
```

To load the allele frequency tables, pass in the `EncodingParameters` object and the CRISPResso2 allele frequency table filenames for each population. For each population, provide a list of filenames corresponding to each replicate.

```
encoding_dataframes = cme.EncodingDataFrames(encoding_parameters=encoding_parameters,
                                                 reference_sequence=AMPLICON,
                                                 population_baseline_filepaths=["CRISPResso_on_sample_baseline_1/Alleles_frequency_table.zip", 
                                                                                "CRISPResso_on_sample_baseline_2/Alleles_frequency_table.zip", 
                                                                                "CRISPResso_on_sample_baseline_3/Alleles_frequency_table.zip"],
                                                 population_target_filepaths=["CRISPResso_on_sample_target_1/Alleles_frequency_table.zip", 
                                                                              "CRISPResso_on_sample_target_2/Alleles_frequency_table.zip", 
                                                                              "CRISPResso_on_sample_target_3/Alleles_frequency_table.zip"],
                                                 population_presort_filepaths=["CRISPResso_on_sample_presort_1/Alleles_frequency_table.zip", 
                                                                               "CRISPResso_on_sample_presort_2/Alleles_frequency_table.zip", 
                                                                               "CRISPResso_on_sample_presort_3/Alleles_frequency_table.zip"],
                                                 wt_filepaths=[root_dir + "CRISPResso_on_sample_wt_1/Alleles_frequency_table.zip"])
```

Run the encoding:
```
encoding_dataframes.read_crispresso_allele_tables()
encoding_dataframes.encode_crispresso_allele_table(progress_bar=True, cores={CPUS})
encoding_dataframes.postprocess_encoding()
```

You can save the results of the encodings to your drive. These files will be used as input to the next modelling step.
```
cme.save_encodings(encoding_dataframes.encodings_collapsed_merged, sort_column="#Reads_presort", filename="./encoding_dataframes_editor_encodings_rep{}.tsv")
cme.save_encodings(encoding_dataframes.population_wt_encoding_processed, sort_column="#Reads_wt", filename="./encoding_dataframes_wt_encodings_rep{}.tsv")
cme.save_encodings_df(encoding_dataframes.population_baseline_encoding_processed, filename="./encoding_dataframes_baseline_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_target_encoding_processed, filename="./encoding_dataframes_target_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_presort_encoding_processed, filename="./encoding_dataframes_presort_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_wt_encoding_processed, filename="./encoding_dataframes_wt_encodings_rep{}.pkl")
```

## Perform modelling of the encoded dataset (in construction)

```
from crispr_millipede import encoding as cme
from crispr_millipede import modelling as cmm

design_matrix_spec = cmm.MillipedeDesignMatrixProcessingSpecification(
    wt_normalization=False,
    total_normalization=True,
    sigma_scale_normalized=True,
    decay_sigma_scale=True,
    K_enriched=5,
    K_baseline=5,
    a_parameter=0.0005
)

millipede_model_specification_set = {
    "joint_replicate_per_experiment_models" : cmm.MillipedeModelSpecification(
        model_types=[cmm.MillipedeModelType.NORMAL_SIGMA_SCALED],
        replicate_merge_strategy=cmm.MillipedeReplicateMergeStrategy.COVARIATE,
        experiment_merge_strategy=cmm.MillipedeExperimentMergeStrategy.SEPARATE,
        S = 5,
        tau = 0.01,
        tau_intercept = 0.0001,
        cutoff_specification=cmm.MillipedeCutoffSpecification(
            per_replicate_each_condition_num_cutoff = 0, 
            per_replicate_all_condition_num_cutoff = 1, 
            all_replicate_num_cutoff = 0, 
            all_experiment_num_cutoff = 0,
            baseline_pop_all_condition_each_replicate_num_cutoff = 3,
            baseline_pop_all_condition_acceptable_rep_count = 2,
            enriched_pop_all_condition_each_replicate_num_cutoff = 3,
            enriched_pop_all_condition_acceptable_rep_count = 2,
            presort_pop_all_condition_each_replicate_num_cutoff = 3,
            presort_pop_all_condition_acceptable_rep_count = 2
        ),
        design_matrix_processing_specification=design_matrix_spec
    )
}
```
