# ![PyPI - Version](https://img.shields.io/pypi/v/crispr-millipede) CRISPR-Millipede User Documentation
<img src="https://github.com/user-attachments/assets/d54fbe8f-2c0e-4354-a209-eab031c3bd64" alt="CRISPR-Millipede logo" width="500"></img> 

# 
 
**CRISPR-Millipede** was developed by the *Pinello Lab* as an easy-to-use Python package for <ins> processing targeted amplicon-sequencing of tiled sequences from base-editing tiling screens to identify functional nucleotides</ins>. By providing amplicon-sequencing of installed alleles from multiple phenotypic populations, CRISPR-Millipede identifies the single-variants that contribute to differences in phenotype. See [this preprint](https://www.biorxiv.org/content/10.1101/2024.09.09.612085v1) for more information on this method! It is expected that you are familiar with Python, command-line tools, and CRISPR screens to follow this guide.

 
  
**Sections**
- [Notes on Experimental Design and Expected Inputs](#notes-on-experimental-design-and-expected-inputs)
- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Instructions](#instructions)
  -  [STEP 1: Run CRISPResso2 to generate allele tables](#step-1-run-crispresso2-to-generate-allele-tables)
  -  [STEP 2: Encode the CRISPResso2 outputs into matrices](#step-2-encode-the-crispresso2-outputs-into-matrices)
  -  [STEP 3: Perform modelling of the encoded dataset](#step-3-perform-modelling-of-the-encoded-dataset)
  -  [STEP 4: Visualization using boardplots](#step-4-generate-board-plots)
  -  [STEP 5: PyDESEQ2 allelic analysis](#step-5-PyDESEQ-based-analysis)
  
### Notes on Experimental Design and Expected Inputs
*Skip this and scroll further down if interested in the tool usage*
- This tool is best used for pooled CRISPR saturation mutagenesis screens of a single focused region. 
- The length of the mutagenized region depends on the desired sequencing read length (i.e. paired-end 150bp sequencing has a max mutagenesis length of 300bp, however, it is desired that there is as much overlap of the paired-ends to maximize sequencing quality). You will perform targeted amplicon-sequencing of your intended mutagenized region. Ensure that no editing occurs at the primer binding sites, and ensure that the primers are tested and optimized beforehand (i.e. difficult-to-amplify or difficult-to-sequence regions may not be suitable for this method, therefore it is essential that this is tested prior to screening).
- It is suggested that you also perform the standard sequencing of the guide RNA to calculate guide RNA enrichment scores in tandem. Therefore, you will split your genomic DNA into two different library preparation approaches: guide RNA sequencing and the aforementioned direct target sequencing.
- The type of mutagenesis is best suited to single-nucleotide mutagenesis (i.e. base-editing and prime-editing). The method has not been extensively tested on in-del mutagenesis. 
- This model was developed and tested on FACS-sorted based screens rather than proliferation screens, however the model may still work for proliferation screens by comparing samples between two separate timepoints.
- Ensure that you have sufficient cell coverage for sequencing, especially if doing both guide RNA and direct target sequencing. You should preferably have roughly 1000 cells * number of guide RNAs in your library for EACH guide RNA and direct sequencing approach (therefore 2000 cells * number of guide RNAs if doing both sequencing approaches) for EACH sample. The cell coverage depends on the editing efficiency and the expected effect sizes. Typically, the sorted population with the phenotypic change from the baseline after perturbation will have the lowest coverage, therefore you should ensure that you have sufficient cell counts in all populations prior to sequencing (by modifying your FACS gates while still maintaining separation between your negative and positive control gRNAs or by simply increasing input cell amount at the expense of longer sort time).
- Ensure that you have sufficient biological replicates (at least 3 replicates).
- It is not necessary to haploidize your region to have single-copy alleles, though this may reduce the noise of the phenotypic scores for each sequenced allele due to certain homozygosity of the sequenced allele. 
- While this method is robust to biases in different editing efficiencies among your guide RNAs since alleles are directly sequenced, ensuring high editing efficiency will increase the per-allele coverage in your samples thereby reducing the necessary cell coverage and increasing statistical power.

See **Figure a** below for a schematic of the experimental design:

  <img src="https://github.com/user-attachments/assets/6ec0a352-aeb2-453b-81d4-ab812c88399b" alt="CRISPR-CLEAR framework" width="300"></img>
    
  <em>**Figure a:** The workflow illustrates the key steps from guide RNA design to data analysis. First, cells stably expressing a base editor are transduced with a library of guide RNAs tiling the regulatory sequence. After editing, cells are FACS-sorted based on the expression of the target protein. Genomic DNA is extracted from sorted cells. Next-generation libraries are prepared to quantify sgRNA counts and to measure the distribution of edits at the endogenous sequence in the sorted population of cells. The left pathway shows the standard approach using sgRNA count-based readout and the CRISPR-SURF pipeline for deconvolution of functional regions. The right pathway depicts the CRISPR-CLEAR approach using direct allele-based readout and the CRISPR-Millipede pipeline, enabling precise genotype-to-phenotype linkage through per-allele and per-nucleotide analysis.</em>

After performing the screen, you should have targetted amplicon-sequencing FASTQs for each of your phenotypic populations (i.e. different FACS gates along with the pre-sort sample) for multiple biological replicates. An overview of the pipeline is to 1) first quality-control using FASTQC to ensure sufficient read quality of all samples, 2) run all the samples through CRISPResso2 to characterize the introduced alleles in your samples, 3) encode the alleles in a numerical representation for Millipede modelling 4) and lastly perform the Millipede modelling to attain your results. See **Figure b** below for a schematic of the pipeline steps:

<img src="https://github.com/user-attachments/assets/0cbb44c8-e073-44c3-be54-fa6239871895" alt="CRISPR-CLEAR framework" width="300"></img>

<em>**Figure b:** Schematic of CRISPR-Millipede workflow.</em>

### Installation

CRISPResso2 is required for first step (a *Pinello Lab* tool), to prepare the input for CRISPR-Millipede. See the [CRISPResso2 repository](https://github.com/pinellolab/CRISPResso2) for installation instructions. You can install this in a different conda environment than CRISPR-Millipede (Preferred). If you want it in the same environment install CRISRPresso2 before CRISPR-Millipede. 

CRISPR-Millipede requires **Python versions >=3.10,<3.12** which can be installed from the [Python download page](https://www.python.org/downloads/) or via **Conda** (see installation of Conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)). Optionally, can use [**mamba**](https://github.com/mamba-org/mamba/blob/main/README.md) for faster installation. For installing Python via Conda:

```conda install python=3.10```.

Additionally, CRISPR-Millipede requires the **PyTorch**, which can be installed via **Conda**. If your computer does not have a CPU, install the CPU-version of PyTorch:

```conda install pytorch```

If you have a GPU, ensure that you have CUDA installed by checking the CUDA version (for example version 11.8):

```nvcc --version```

If you don't have CUDA installed, follow the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Then, install the appropriate GPU version of PyTorch with the correct version of the **pytorch-cuda** based on the CUDA version installed on your OS (for example version 11.8):

```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```

Once you have all Python and PyTorch dependencies installed, CRISPR-Millipede can easily be installed from PyPi which should only take a few minutes. PIP will ensure that all Python package dependencies are installed:

```pip install crispr-millipede==0.1.97```, 

***Did you also directly sequence your guide RNAs?*** It is recommended you do so to compare against the CRISPR-Millipede results from target amplicon-sequencing. You could map your guide sequences using tools from the *Pinello Lab* such as [CRISPR-Correct](https://github.com/pinellolab/CRISPR-Correct) and analyze the resulting counts using [CRISPR-SURF](https://github.com/pinellolab/CRISPR-SURF/tree/master) as done in the original paper! 

PyDESeq2 can also be installed from PyPi, using the following command:

```pip install pydeseq2```

### System Requirements
CRISPR-Millipede can run on [any operating system where Python versions >=3.10,<3.12 can be installed](https://www.python.org/downloads/operating-systems/) and where [PyTorch can be installed](https://pytorch.org/get-started/locally/). To speed up model performance, CRISPR-Millipede can utilize both CPUs (for multi-threading) and GPUs (for model training) and is highly recommended to allow the pipeline to run in the span of a couple hours, though the tool can still work on single core non-GPU computers but may run in the span of a day for each run attempt depending on the FASTQ sizes. 

### Installation and Run Time
On a Macbook Pro (M2 Chip with 32 GB ram)
- Installation takes about 1 min 20 secs via pip after installing PyTorch
- Running Step 1 (CRISPResso2) takes about 5 mins on sg218 example
- Running Step 2 (Encoding) takes about 20 mins on sg218 example
- Running Step 3 (Millipede: model_run = cmm.MillipedeModelExperimentalGroup(experiments_inputdata=model_input_data, device=cmm.MillipedeComputeDevice.CPU) takes about 2 minutes for the sg218 example in the notebook


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

Below contains the class definition (and default values) of the `EncodingParameters` that you will need to instantiate:

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

cme.save_encodings(encoding_dataframes.encodings_collapsed_merged, sort_column="#Reads_presort", filename=prefix_label + "encoding_dataframes_editor_encodings_rep{}.tsv")
cme.save_encodings(encoding_dataframes.population_wt_encoding_processed, sort_column="#Reads_wt", filename=prefix_label + "encoding_dataframes_wt_encodings_rep{}.tsv")
cme.save_encodings_df(encoding_dataframes.population_baseline_encoding_processed, filename=prefix_label + "encoding_dataframes_baseline_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_target_encoding_processed, filename=prefix_label + "encoding_dataframes_target_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_presort_encoding_processed, filename=prefix_label + "encoding_dataframes_presort_editor_encodings_rep{}.pkl")
cme.save_encodings_df(encoding_dataframes.population_wt_encoding_processed, filename=prefix_label + "encoding_dataframes_wt_encodings_rep{}.pkl")
```

### STEP 3: Perform modelling of the encoded dataset
*Now that we have the encoded representation of the alleles, we will now perform Millipede modelling off of this representation. For documentation on the Millipede model sub-package, see [here](https://millipede.readthedocs.io/en/latest/getting_started.html).*

**Set the model parameters:** Below contains the class definition (and default values) of the `MillipedeDesignMatrixProcessingSpecification` that you will need to instantiate:

```
@dataclass
class MillipedeDesignMatrixProcessingSpecification:
    wt_normalization: bool = True # Normalize the read count base on the unedited allele counts
    total_normalization: bool = False # Normalize the read count based on the total sum of all allele counts
    sigma_scale_normalized: bool = False # If using the NormalLikelihoodVariableSelector, determine if the sigma_scale factor will be based on the normalized read count
    decay_sigma_scale: bool = True # Set the sigma_scale factor based on the decay function
    K_enriched: Union[float, List[float], List[List[float]]] = 5 # Set the K_enriched value of the decay function
    K_baseline: Union[float, List[float], List[List[float]]] = 5 # Set the K_baseline value of the decay function
    a_parameter: Union[float, List[float], List[List[float]]] = 300 # Set the a_parameter of the decay function
```

Additionally, you will need to specify the type of model as well. Below contains the class definition (and default values) of the `MillipedeModelSpecification` that you will need to instantiate:

```
@dataclass
class MillipedeModelSpecification:
    """
        Defines all specifications to produce Millipede model(s)
    """
    model_types: List[MillipedeModelType] 
    replicate_merge_strategy: MillipedeReplicateMergeStrategy
    experiment_merge_strategy: MillipedeExperimentMergeStrategy
    cutoff_specification: MillipedeCutoffSpecification
    design_matrix_processing_specification: MillipedeDesignMatrixProcessingSpecification
    shrinkage_input: Union[MillipedeShrinkageInput, None] = None
    S: float = 1.0 #S parameter
    tau: float = 0.01 #tau parameter
    tau_intercept: float = 1.0e-4
```

There are sub-classes you will need to instantiate. For instance, the `MillipedeReplicateMergeStrategy` specifies how multiple replicates are handled during modelling:
```
class MillipedeReplicateMergeStrategy(Enum):
    """
        Defines how separate replicates will be treated during modelling
    """
    SEPARATE = "SEPARATE" # Replicates are modelled separately; one model per replicate
    SUM = "SUM" # (Normalized) counts for all replicates are summed together; one model for all replicates
    COVARIATE = "COVARIATE" # Replicates are jointly modelled, though replicate ID is included in the model design matrix 
```

*Recommended to run one version in `MillipedeReplicateMergeStrategy.SEPARATE` to assess individual replicate consistency, then if successful, run a final model in `MillipedeReplicateMergeStrategy.COVARIATE`*
    
Likewise, the `MillipedeExperimentMergeStrategy` specifies how multiple experiments (i.e. screens with different editors) are handled during modelling.
```
class MillipedeExperimentMergeStrategy(Enum):
    """
        Defines how separate experiments will be treated during modelling
    """
    SEPARATE = "SEPARATE"
    SUM = "SUM"
    COVARIATE = "COVARIATE"
```

The `MillipedeModelType` specifies what likelihoood function to use for model fitting. See the [Millipede documentation](https://millipede.readthedocs.io/en/latest/selection.html) for more information. 
```
class MillipedeModelType(Enum):
    """
        Defines the Millipede model likelihood function used
    """
    NORMAL = "NORMAL"
    NORMAL_SIGMA_SCALED = "NORMAL_SIGMA_SCALED"
    BINOMIAL = "BINOMIAL"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"
```
*We recommend using the NORMAL_SIGMA_SCALED model, you will need to define the K_enriched, K_baseline, a, and decay_sigma_scale paramters to specify how the sigma_scale_factor is calculated.*

Here is an example of specifying the complete input parameters for modelling:
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
    "model_specification_1" : cmm.MillipedeModelSpecification(
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

**Load in the encoding data:** Now that you have specified the model inputs, let's load the encoding data in, which should be straightforward:

```
prefix_label ="20240916_v1_example_"
encoding_filename = prefix_label + "encoding_dataframes_editor_encodings_rep{}.tsv"

# This will load in the data
model_input_data = cmm.MillipedeInputDataExperimentalGroup(
    data_directory="./", 
    enriched_pop_fn_experiment_list = [encoding_filename],
    enriched_pop_df_reads_colname = "#Reads_target",
    baseline_pop_fn_experiment_list = [encoding_filename],
    baseline_pop_df_reads_colname = "#Reads_baseline", 
    presort_pop_fn_experiment_list = [encoding_filename],
    presort_pop_df_reads_colname = '#Reads_presort',
    experiment_labels = ["editor"],
    reps = [0,1,2],
    millipede_model_specification_set = millipede_model_specification_set
   )
```

**Run the model:** Now that you have specified the inputs, we will now run the model. You have the option to use the CPU or GPU for modelling.

```
model_run = cmm.MillipedeModelExperimentalGroup(experiments_inputdata=model_input_data, device=cmm.MillipedeComputeDevice.GPU)
```

**Explore the results:** The model will provide posterior inclusion probabilities (PIP) and beta coefficient scores for each feature/variant that was included in the model and not filtered out during the encoding step:

```
beta_df = paired_end_experiments_models_denoised.millipede_model_specification_set_with_results['model_specification_1'].millipede_model_specification_result_input[0].millipede_model_specification_single_matrix_result[cmm.MillipedeModelType.NORMAL_SIGMA_SCALED].beta
pip_df = paired_end_experiments_models_denoised.millipede_model_specification_set_with_results['model_specification_1'].millipede_model_specification_result_input[0].millipede_model_specification_single_matrix_result[cmm.MillipedeModelType.NORMAL_SIGMA_SCALED].pip
sigma_hit_table = paired_end_experiments_models_denoised.millipede_model_specification_set_with_results["joint_replicate_per_experiment_models"].millipede_model_specification_result_input[0].millipede_model_specification_single_matrix_result[cmm.MillipedeModelType.NORMAL_SIGMA_SCALED].summary

sigma_hit_table.to_csv('MillipedeOutput.csv', index=True)

```
**Model Output Table:** The output table (sigma_hit_table) will look like this where for each covariate you are given a PIP, Beta, Conditional PIP, and Conditional Beta

<img width="721" alt="Screenshot 2024-10-08 at 5 17 55 PM" src="https://github.com/user-attachments/assets/9b946fc2-c7dd-43c6-98e7-4bf90864de01">

### STEP 4: Generate Board Plots

**Board Plots:** Board Plots can be generated by using the board plot function provided in CRISPR-Millipede. Board Plots require the millipede table, presort, and wt editing frequencies which can be generated using the functions below. 

```
paired_merged_raw_encodings = cmm.RawEncodingDataframesExperimentalGroup().read_in_files_constructor(
    enriched_pop_fn_encodings_experiment_list = ["./encoding_dataframes_target_editor_encodings_rep{}.pkl"],
    baseline_pop_fn_encodings_experiment_list = ["./encoding_dataframes_baseline_editor_encodings_rep{}.pkl"],
    presort_pop_fn_encodings_experiment_list = ["./encoding_dataframes_presort_editor_encodings_rep{}.pkl"],
    experiment_labels = ["ABE8e"],
    ctrl_pop_fn_encodings="./encoding_dataframes_wt_editor_encodings_rep{}.pkl",
    ctrl_pop_labels="WT",
    reps = [0,1,2],
   )
paired_merged_raw_encodings_editing_freqs.presort_pop_encoding_editing_per_variant_freq_avg[0].to_csv('presort_editing_freqs_avg_editor.csv')
paired_merged_raw_encodings_editing_freqs.baseline_pop_encoding_editing_per_variant_freq_avg[0].to_csv('baseline_editing_freqs_avg_editor.csv')
paired_merged_raw_encodings_editing_freqs.enriched_pop_encoding_editing_per_variant_freq_avg[0].to_csv('target_editing_freqs_avg_editor.csv')
paired_merged_raw_encodings_editing_freqs.ctrl_pop_encoding_editing_per_variant_freq_avg[0].to_csv('wt_editing_freqs_avg_editor.csv')

cmm.plot_millipede_boardplot(editorName (ABE8e or evoCDA), 'MillipedeOutput.csv', 'presort_editing_freqs_avg_editor.csv' , 'wt_editing_freqs_avg_editor.csv', start,end, AMPLICON, outputPath = "Boardplot.svg")

```
<img width="668" alt="Screenshot 2024-10-09 at 2 37 18 PM" src="https://github.com/user-attachments/assets/a698298c-3d54-49b6-b94b-cdf3c6d329e4">

### STEP 5: PyDESeq2 based analysis
The encoded representation of the alleles can also be fed into PyDESeq2, to calculate the differential distribution of each allele across the sorted populations. For documentation on PyDESeq2, see [here](https://pydeseq2.readthedocs.io/en/latest/index.html#).

PyDESeq2 takes in a count and design matrix, along with several parameters:

```
inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=count_df,
    metadata=metadata_df,
    design_factors="condition",
    refit_cooks=True,
    inference=inference,
    # n_cpus=8, # n_cpus can be specified here or in the inference object
)
```
**See [notebooks/STEP5_ABE8e_DESeq2_Demo.ipynb](https://github.com/pinellolab/CRISPR-millipede-target/blob/master/notebooks/STEP5_ABE8e_DESeq2_Demo.ipynb) for instructions on how to format the input matrices and run PyDESeq2.**

After running pyDESeq2, we can visualize a volcano plot of the per-allele scores derived through the model:

```
def contains_edit_special(edit, edit2):
    colors = []
    sizes = []
    
    subset_df = results_df.copy()
    
    for index, row in subset_df.iterrows():
        if len(set(edit).intersection(set(index.split(",")))) > 0:
            colors.append("#00AEEF")
            sizes.append(40)
        elif len(set(edit2).intersection(set(index.split(",")))) > 0:
            colors.append("#EC008C")
            sizes.append(40)
        else:
            colors.append("gray")
            sizes.append(40)
            subset_df.drop(index, inplace=True)
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    
    # Scatter plot
    plt.scatter(results_df['log2FoldChange'] * -1, 
                results_df['-10 * log(pvalue)'],
                c=colors, s=sizes, alpha=0.3)
    
    # Set x-axis to log2 scale
    plt.xscale('symlog', base=2)
    
    # Set axis labels and title
    plt.xlabel("Log2 Fold Change [CD19+ vs CD19-]", fontsize=14)
    plt.ylabel("-10 * log10(pvalue)", fontsize=14)
    plt.title("Volcano Plot", fontsize=16)
    
    # Set x-axis limits and ticks
    plt.xlim(-10, 10)
    
    # Set y-axis limits
    plt.ylim(0, 30)
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the figure
    plt.savefig("ABE8e_allelic_analysis_w_MillipedeHits.svg")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    # Display the subset dataframe
    display(subset_df)
```

The parameters "edit1" and "edit2" can be used to selectively color alleles that exhibit certain sets of edits:

```
contains_edit_special(["223A>G", "230A>G"], ["151A>G"])
```

![image](https://github.com/user-attachments/assets/32c9451a-bf65-45f4-bf2c-87317ef920fa)
