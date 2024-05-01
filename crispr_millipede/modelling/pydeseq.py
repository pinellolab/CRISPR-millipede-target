from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data
import pandas as pd

from typing import List, Optional

from functools import reduce

def run_pydeseq2(encoding_reps: List[pd.DataFrame], enriched_pop_df_reads_colname: str, baseline_pop_df_reads_colname: str, start: Optional[int] = None, end: Optional[int] = None):
    nucleotide_ids = [col for col in encoding_reps[0].columns if ">" in col]
    non_nucleotide_ids = encoding_reps[0].loc[:, [col not in nucleotide_ids for col in encoding_reps[0]]].columns

    # Rename non-feature columns to add merged suffixed based on replicate number
    encoding_reps = [encoding_rep.rename(columns=dict(zip(non_nucleotide_ids,[col + f"_{index}" for col in non_nucleotide_ids]))) for index, encoding_rep in enumerate(encoding_reps)]
    merged_encoding_reps = reduce(lambda left,right: pd.merge(left, right, on=nucleotide_ids, how='outer').fillna(0), encoding_reps)
    # Subset the columns to only the count columns for PyDESeq2

    sample_names = [colname for index in range(len(encoding_reps)) for colname in [enriched_pop_df_reads_colname + f"_{index}", baseline_pop_df_reads_colname + f"_{index}"]]
    condition_names = [colname for index in range(len(encoding_reps)) for colname in [enriched_pop_df_reads_colname, baseline_pop_df_reads_colname]]
    replicate_names = [colname for index in range(len(encoding_reps)) for colname in [index, index]]

    metadata_df = pd.DataFrame({"sample": sample_names, "condition":condition_names, "replicate_names": replicate_names})
    metadata_df = metadata_df.set_index("sample")

    merged_encoding_reps_count = merged_encoding_reps.loc[:, sample_names]
    merged_encoding_reps_count = merged_encoding_reps_count.T

    # TODO 20240422: See if DESeq handles normalization, if they do, use raw counts instead of casting to int
    merged_encoding_reps_count = merged_encoding_reps_count.astype(int)

    # NOTE 20240424: Only just using CPU=1 since not intensive. If it is bottlneck, can set cpu's based on availble cores
    inference = DefaultInference(n_cpus=1)
    dds = DeseqDataSet(
        counts=merged_encoding_reps_count,
        metadata=metadata_df,
        design_factors="condition",
        refit_cooks=True,
        inference=inference,
        # n_cpus=8, # n_cpus can be specified here or in the inference object
    )

