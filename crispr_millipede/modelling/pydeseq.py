from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data
import pandas as pd

from typing import List, Optional

from functools import reduce
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

def run_pydeseq2(encoding_reps: List[pd.DataFrame], enriched_pop_df_reads_colname: str, baseline_pop_df_reads_colname: str, start: Optional[int] = None, end: Optional[int] = None, amplicon: Optional[str] = None) -> pd.DataFrame:

    # Get the nucleotide IDs from the per-replicate feature dataframes
    nucleotide_ids = [col for col in encoding_reps[0].columns if ">" in col]
    non_nucleotide_ids = [col for col in encoding_reps[0].columns if ">" not in col]
    
    # Trim nucleotide IDs by given start/end
    if ((start is not None) and (end is not None)):
        nucleotide_ids = [nt_id for nt_id in nucleotide_ids if int(nt_id[:nt_id.find(">")-1]) in range(start, end)]
        encoding_reps = [encoding_rep.loc[:, [*nucleotide_ids, *non_nucleotide_ids]] for encoding_rep in encoding_reps] # Filter out nucleotides not in range

    # Rename non-feature columns to add merged suffixed based on replicate number (so that we can merge all replicate dataframes together without column naming conflicts)
    encoding_reps = [encoding_rep.rename(columns=dict(zip(non_nucleotide_ids,[col + f"_{index}" for col in non_nucleotide_ids]))) for index, encoding_rep in enumerate(encoding_reps)]
    merged_encoding_reps = reduce(lambda left,right: pd.merge(left, right, on=nucleotide_ids, how='outer').fillna(0), encoding_reps)
    merged_encoding_reps = merged_encoding_reps.groupby(nucleotide_ids, as_index=False).sum()
    
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
    stat_res.lfc_shrink(coeff=f"condition_{enriched_pop_df_reads_colname_replaced}_vs_{baseline_pop_df_reads_colname_replaced}")

    # Add pyDESeq2 columns to new design matrix
    merged_encoding_reps["score"] = stat_res.results_df.loc[:, "log2FoldChange"].values
    merged_encoding_reps["scale_factor"] = stat_res.results_df.loc[:, "lfcSE"].values
    merged_encoding_reps["pvalue"] = stat_res.results_df.loc[:, "pvalue"].values
    merged_encoding_reps["padj"] = stat_res.results_df.loc[:, "padj"].values

    # Set index as allele
    if amplicon is not None:
        def retrieve_allele_id(row):
            nucleotide_ids = [col for col in row.index if ">" in col] 
            row_restricted = row[nucleotide_ids]
            cols_selected = row_restricted[row_restricted == 1].index.tolist()

            allele = amplicon
            for col in cols_selected:
                demarker = col.find(">")
                pos = int(col[:demarker - 1])
                alt = col[demarker+1:]
                allele = allele[:pos] + alt + allele[pos + 1:]
            
            # Trim allele if start/end provided
            allele = allele[start:end] if ((start is not None) and (end is not None)) else allele

            return allele

        merged_encoding_reps.index = merged_encoding_reps.apply(retrieve_allele_id, axis=1)
        
    


    # Send back final result
    return merged_encoding_reps

    


def visualize_deseq2_result(merged_encoding_reps, title_label, score_col="score", pvalue_col="pvalue", filename=None):
    fig = go.Figure()
    scatter = go.Scatter(
        x=merged_encoding_reps[score_col],
        y=-np.log10(merged_encoding_reps[pvalue_col]),
        mode="markers",
        marker=dict(size=6, opacity=0.5),
        hovertext=list(merged_encoding_reps.index)
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title=title_label, title_font=dict(size=18), range=[-2.5, 2.5], showgrid = False, zeroline = False, ticks="outside", ticklen = 10, tickfont=dict(size=15)),
        yaxis=dict(title="-log10(pvalue)", title_font=dict(size=18), range=[-10, 200], showgrid = False, zeroline = False, ticks="outside", ticklen = 10, tickfont=dict(size=15)),
        plot_bgcolor="white"
    )
    
    fig.add_trace(scatter)
    
    fig.add_annotation(
        x=1, y=1,
        xref="paper", yref="paper",
        text="Total number of alleles: " + str(merged_encoding_reps.shape[0]),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(size=14, color="black")
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black")

    if filename:
        fig.write_image(filename)
    else:
        fig.show()
