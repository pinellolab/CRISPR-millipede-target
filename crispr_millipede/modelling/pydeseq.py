from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data
import pandas as pd

from typing import List, Optional

from functools import reduce

def run_pydeseq2(encoding_reps: List[pd.DataFrame], enriched_pop_df_reads_colname: str, baseline_pop_df_reads_colname: str, start: Optional[int] = None, end: Optional[int] = None) -> pd.DataFrame:

    # Get the nucleotide IDs from the per-replicate feature dataframes
    nucleotide_ids = [col for col in encoding_reps[0].columns if ">" in col]
    non_nucleotide_ids = encoding_reps[0].loc[:, [col not in nucleotide_ids for col in encoding_reps[0]]].columns

    # Rename non-feature columns to add merged suffixed based on replicate number (so that we can merge all replicate dataframes together without column naming conflicts)
    encoding_reps = [encoding_rep.rename(columns=dict(zip(non_nucleotide_ids,[col + f"_{index}" for col in non_nucleotide_ids]))) for index, encoding_rep in enumerate(encoding_reps)]
    merged_encoding_reps = reduce(lambda left,right: pd.merge(left, right, on=nucleotide_ids, how='outer').fillna(0), encoding_reps)
    
    # Subset the columns to only the count columns for the PyDESeq2 input
    sample_names = [colname for index in range(len(encoding_reps)) for colname in [enriched_pop_df_reads_colname + f"_{index}", baseline_pop_df_reads_colname + f"_{index}"]]
    condition_names = [colname for index in range(len(encoding_reps)) for colname in [enriched_pop_df_reads_colname, baseline_pop_df_reads_colname]]
    replicate_names = [colname for index in range(len(encoding_reps)) for colname in [index, index]]

    # Prepare metadata DF for DESeq2
    metadata_df = pd.DataFrame({"sample": sample_names, "condition":condition_names, "replicate_names": replicate_names})
    metadata_df = metadata_df.set_index("sample")

    merged_encoding_reps_count = merged_encoding_reps.loc[:, sample_names]
    merged_encoding_reps_count = merged_encoding_reps_count.T

    merged_encoding_reps_count = merged_encoding_reps_count.astype(int)

    # NOTE 20240424: Only just using CPU=1 since not intensive. If it is bottlneck, can set cpu's based on availble cores
    # Prepare DESeq2 function
    inference = DefaultInference(n_cpus=1)
    dds = DeseqDataSet(
        counts=merged_encoding_reps_count,
        metadata=metadata_df,
        design_factors="condition",
        refit_cooks=True,
        inference=inference,
        # n_cpus=8, # n_cpus can be specified here or in the inference object
    )

    # Run DESeq2
    dds.deseq2()
    stat_res = DeseqStats(dds, inference=inference)
    stat_res.summary(lfc_null=0.1, alt_hypothesis="greaterAbs")
    #stat_res.plot_MA(s=20)

    # Perform DESeq2 LFC shrinking
    baseline_pop_df_reads_colname_replaced = baseline_pop_df_reads_colname.replace("_","-")
    enriched_pop_df_reads_colname_replaced = enriched_pop_df_reads_colname.replace("_","-")
    stat_res.lfc_shrink(coeff=f"condition_{baseline_pop_df_reads_colname_replaced}_vs_{enriched_pop_df_reads_colname_replaced}")

    # Add pyDESeq2 columns to new design matrix
    merged_encoding_reps["score"] = stat_res.results_df.loc[:, "log2FoldChange"].values
    merged_encoding_reps["scale_factor"] = stat_res.results_df.loc[:, "lfcSE"].values

    # Send back final result
    return merged_encoding_reps

    

