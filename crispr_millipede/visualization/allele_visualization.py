from collections import defaultdict
import pandas as pd
from typing import List
from ..modelling.models_processing import MillipedeInputDataExperimentalGroup

def allelic_analysis(experiments_inputdata: MillipedeInputDataExperimentalGroup, rep:int, start:int, end: int, ref_nt: str, alt_nt: str):
    #setup dataframe
    df_list: List[pd.DataFrame] = experiments_inputdata.millipede_model_specification_set_with_data["per_replicate_per_experiment_models"][1].data[0]
    cols = list(df_list[rep].columns.values) # Contain both the read counts, and the NT features - MultiIndex
    
    keep_list = []
    allele_dict = {}
    edit_dict = defaultdict(list)

    df = df.loc[:, cols.get_level_values("Position").isin(range(start,end)) and (cols.get_level_values("Ref")==ref_nt) and (cols.get_level_values("Alt")==alt_nt)]
    df = df.loc[df.sum(axis=1) >= 1, :]

    for index, row in df_list[rep].iterrows(): # For each row in dataframe... build up alleles from encoding
        keep = False
        allele = ""
        
        for i in range(start, end): # Coordinates
            #look through edit encodings for each position, see if there is an edit
            selected = cols[i*5:(i+1)*5]
            demarker = selected[0].find(">")
            ref = selected[0][demarker - 1:demarker]  
            
            #if edited, get alt nucleotide
            edited = False
            for selec in selected:
                if row[selec] == 1.0 and intended_edit in selec:
                    alt = '<span style="color: red;">' + selec[demarker+1:] + '</span>' # BOLD: "\033[1m" + text + "\033[0m"
                    edit_dict[index].append(selec)
                    edited = True
                    keep = True
                    
            if not edited:
                alt = ref
            #build allele sequence from edit encodings
            allele += alt
        
        if keep:
            keep_list.append(index)
            allele_dict[index] = allele
    
    #only keep edited alleles in the table
    new_df = df_list[rep].loc[keep_list]
    
    edits = []
    alleles = []
    
    #add edit and allele information to table
    for ind in list(new_df.index):
        edito = ", ".join(edit_dict[ind]) # NOTE: For each allele, have list of edits
        edits.append(edito)
        
        allelo = allele_dict[ind]
        alleles.append(allelo)
        
    new_df["edits"] = edits
    new_df["alleles"] = alleles
    
    #parse read columns
    read_col_names = []
    for col in cols:
        if "#Reads" in col or "total_reads" in col:
            read_col_names.append(col)
            
    #setup column names
    new_df["score_std"] = new_df["score"]
    cut_df = new_df[["alleles", "edits", "score", "score_std"] + read_col_names].set_index("alleles")
    cut_df = cut_df.rename(columns = {"score": "score_mean"})
    
    #setup group by operation
    aggregations = {}
    aggregations["score_mean"] = "mean"
    aggregations["score_std"] = "std"
    for col in read_col_names:
        aggregations[col] = "sum"
        
    #perform group by, sort by total reads
    grouped = cut_df.groupby(["alleles", "edits"]).agg(aggregations)
    final = grouped.sort_values("total_reads", ascending = False).style.to_html()
    
    #return formatted table
    return HTML(final)





allelic_analysis(paired_end_experiments_inputdata, rep = 2, start = 219, end = 236, intended_edit = "A>G")