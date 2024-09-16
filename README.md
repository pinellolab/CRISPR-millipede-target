# CRISPR-Millipede

CRISPR-Millipede was developed by the *Pinello Lab* as an easy-to-use Python package for <ins> processing targeted amplicon-sequencing of tiled sequences from base-editing tiling screens to identify functional nucleotides</ins>. By providing amplicon-sequencing of installed alleles from multiple phenotypic populations, CRISPR-Millipede identifies the single-variants that contribute to differences in phenotype. See [this preprint](https://www.biorxiv.org/content/10.1101/2024.09.09.612085v1) for more information on this method! It is expected that you are familiar with Python, command-line tools, and CRISPR screens to follow this guide.


  <img src="https://github.com/user-attachments/assets/6ec0a352-aeb2-453b-81d4-ab812c88399b" alt="CRISPR-CLEAR framework" width="300"></img>
    
  <em>**Figure a:** The workflow illustrates the key steps from guide RNA design to data analysis. First, cells stably expressing a base editor are transduced with a library of guide RNAs tiling the regulatory sequence. After editing, cells are FACS-sorted based on the expression of the target protein. Genomic DNA is extracted from sorted cells. Next-generation libraries are prepared to quantify sgRNA counts and to measure the distribution of edits at the endogenous sequence in the sorted population of cells. The left pathway shows the standard approach using sgRNA count-based readout and the CRISPR-SURF pipeline for deconvolution of functional regions. The right pathway depicts the CRISPR-CLEAR approach using direct allele-based readout and the CRISPR-Millipede pipeline, enabling precise genotype-to-phenotype linkage through per-allele and per-nucleotide analysis.</em>

<img src="https://github.com/user-attachments/assets/0cbb44c8-e073-44c3-be54-fa6239871895" alt="CRISPR-CLEAR framework" width="300"></img>

<em>**Figure b:** Schematic of CRISPR-Millipede workflow.</em>

### Installation
CRISPR-Millipede can be easily installed from PyPi `pip install crispr-millipede`, which should only take a couple minutes. CRISPR-Millipede requires **Python versions >=3.8,<3.12** which can be installed from the [Python download page](https://www.python.org/downloads/). 

You will also need to run CRISPResso2, a *Pinello Lab* tool, to prepare the input for CRISPR-Millipede. See the [CRISPResso2 repository](https://github.com/pinellolab/CRISPResso2) for installation instructions.

***Did you also directly sequence your guide RNAs?*** It is recommended you do so to compare against the CRISPR-Millipede results from target amplicon-sequencing. You could map your guide sequences using tools from the *Pinello Lab* such as [CRISPR-Correct](https://github.com/pinellolab/CRISPR-Correct) and analyze the resulting counts using [CRISPR-SURF](https://github.com/pinellolab/CRISPR-SURF/tree/master) as done in the original paper! 

### System Requirements
CRISPR-Millipede can run on [any operating system where Python versions >=3.8,<3.12 can be installed](https://www.python.org/downloads/operating-systems/). To speed up model performance, CRISPR-Millipede can utilize both CPUs (for multi-threading) and GPUs (for model training) and is highly recommended, though the tool can still work on single core non-GPU computers. 

## Instructions

### STEP 1: Run CRISPResso2 to generate allele tables
*We need to take the raw amplicon-sequencing data and encode it into an input that CRISPR-Millipede accepts. It is suggested that your amplicon-sequencing data is quality-controlled using [FASTQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/) to ensure sequencing quality.*

CRISPR-Millipede's encoding step takes in as input the allele frequency tables produced from [CRISPResso2](https://github.com/pinellolab/CRISPResso2), a *Pinello Lab* tool for processing amplicon-sequencing data from CRISPR experiments. Refer to the [CRISPResso2 documentation](https://github.com/pinellolab/CRISPResso2) for instructions on how to run CRISPResso2, which may depend on the type of CRISPR editing performed in your experiment.


Example command for a base-editing experiment:
```
CRISPRessoBatch
  -bs {FASTQ_FILENAME} -a {AMPLICON_SEQUENCE} -an {AMPLICON_NAME}
  -q {QUALITY}
  --exclude_bp_from_left {EX_LEFT} --exclude_bp_from_right {EX_RIGHT}
  --no_rerun -n {SCREEN_NAME}
  --min_frequency_alleles_around_cut_to_plot 0.001
  --max_rows_alleles_around_cut_to_plot 500
  -p 20 --plot_window_size 4 --base_editor_output -w 0
  -bo {OUTPUT_DIRECTORY}
```

Run CRISPResso2 for all samples and replicates. For each sample, CRISPResso2 will produce an allele frequency table named "Alleles_frequency_table.zip" which is used as input to the CRISPR-Millipede package. You will need these files in the next step. CRISPResso2 will also produce several other plots characterizing the editing patterns of your samples which will be useful for initial exploration of your data prior to modelling!

### STEP 2: Encode the CRISPResso2 outputs into matrices
*The CRISPResso2 output contains a table of alleles and their read counts for each sample. The alleles are represented as strings, though the strings must be encoded into a numerical representation for CRISPR-Millipede modelling.*

Import and prepare the parameters of the encoding step by passing in the amplicon sequence (required), the acceptable variant types (optional), predicted editing sites (optional), population colummn suffixes for indexing (required), and encoding edge trimming for reducing sequencing background (optional) to the `EncodingParameters` class.

Below contains the class definition (and default values) of the EncodingParameters that you will need to instantiate:

```
@dataclass
class EncodingParameters:
    complete_amplicon_sequence: str # Amplicon sequence string
    population_baseline_suffix: Optional[str] = "_baseline" # Typically the population that unedited cells are primarily in. Suffix label
    population_target_suffix: Optional[str] = "_target" # The population used to calculate variant enrichment relative to the baseline population. Suffix label
    population_presort_suffix: Optional[str] = "_presort" # The un-sorted population used to calculate total editing efficiencies. Suffix label
    wt_suffix: Optional[str] = "_wt" # An unedited population to calculate the sequencing error background. Suffix label
    guide_edit_positions: List[int] = field(default_factory=list) # Position of expected editing sites. Positions are relative to the amplicon sequence. 0-based.
    guide_window_halfsize: int = 3 # Expected editing window size. Only edits in range(guide_edit_position-guide_window_halfsize,guide_edit_position+guide_window_halfsize+1) for all positions will be considered for modelling
    minimum_editing_frequency: float = 0 # Frequency of variants to consider for editing, may be useful for removing sequencing background.
    minimum_editing_frequency_population: List[str] = field(default_factory=list) # Population to consider for removal of variants by frequency, i.e. ["presort"]
    variant_types: List[Tuple[str, str]] = field(default_factory=list) #  List of variants to consider for modelling. Variants represented as two-value tuple where first index is REF and second index is ALT. i.e.  [("A", "G"), ("T", "C")] for adenine base-editing variants.
    trim_left: int = 0 # Filtering positions on left side of amplicon
    trim_right: int = 0 # Filtering positions on right side of amplicon
    remove_denoised: bool = False # Remove filtered features (from above criteria) from model input.  
```


Example of setting encoding parameters:
```
from crispr_millipede import encoding as cme

AMPLICON = "ACTGACTGACTGACTGACTGACTG" # Put your complete reference amplicon-sequence here
ABE_VARIANT_TYPES = [("A", "G"), ("T", "C")] # Optional: If using an adenine base-editor
CBE_VARIANT_TYPES = [("C", "T"), ("G", "A")] # Optional: If using a cytosine base-editor
encoding_parameters = cme.EncodingParameters(complete_amplicon_sequence=AMPLICON,
                            population_baseline_suffix="_baseline", 
                            population_target_suffix="_target", 
                            population_presort_suffix="_presort", 
                            wt_suffix="_wt", 
                            trim_left=20, 
                            trim_right=20, 
                            variant_types=ABE_VARIANT_TYPES, 
                            remove_denoised=True)
```


To load the CRISPResso2 allele frequency tables into CRISPR-Millipede from STEP 1, pass in the `EncodingParameters` object and the CRISPResso2 allele frequency table filenames from STEP 1 for each population. For each population, provide a list of filenames corresponding to each replicate:

```
encoding_dataframes = cme.EncodingDataFrames(encoding_parameters=encoding_parameters, #  From example above
                                                 reference_sequence=encoding_parameters.complete_amplicon_sequence,
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


Perform the encoding:
```
encoding_dataframes.read_crispresso_allele_tables() # This reads in the CRISPResso2 table
encoding_dataframes.encode_crispresso_allele_table(progress_bar=True, cores={CPUS}) # Performs the initial encoding. Replace {CPUs} with the number of CPUs for parallelization on your system. 
encoding_dataframes.postprocess_encoding() # Postprocesses the encoding with the filtering criteria from above.
```

Highly suggested to save the results of the encodings to your drive. Encouraged to include a prefix to version the results. These files will be used as input to the next modelling STEP 3.
```
prefix_label ="20240916_v1_example_"

cme.save_encodings(encoding_dataframes.encodings_collapsed_merged, sort_column="#Reads_presort", filename="./encoding_dataframes_editor_encodings_rep{}.tsv")
cme.save_encodings(encoding_dataframes.population_wt_encoding_processed, sort_column="#Reads_wt", filename=prefix_label + "encoding_dataframes_wt_encodings_rep{}.tsv")
cme.save_encodings_df(encoding_dataframes.population_baseline_encoding_processed, filename=prefix_label + "encoding_dataframes_baseline_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_target_encoding_processed, filename=prefix_label + "encoding_dataframes_target_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_presort_encoding_processed, filename=prefix_label + "encoding_dataframes_presort_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_wt_encoding_processed, filename=prefix_label + "encoding_dataframes_wt_encodings_rep{}.pkl")
```

### STEP 3: Perform modelling of the encoded dataset (in construction)

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
