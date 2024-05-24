import pandas as pd
import numpy as np
from pandarallel import pandarallel
from dataclasses import dataclass
from typing import List, Optional, Tuple
from functools import reduce
import copy

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

''' 
    Function to perform substitution encoding
'''
def get_substitution_encoding(aligned_sequence, original_seq, skip_index=0):
    assert len(aligned_sequence) == len(original_seq) # Ensure the aligned sequence (from allele table) is equal size to the reference sequence
    

    nucleotides = ["A","C","T","G", "N", "-"] # List of possible nucleotides
    encodings_per_position = []
    mismatch_mappings_per_position = []
    for index in range(0, len(original_seq)): # Iterate through each base and check for substitution
        # TODO Ensure sequences are uppercase
        # Create array with possible mismatches
        nucleotides_mm = nucleotides[:]
        nucleotides_mm.remove(original_seq[index])

        mm_encoding = pd.Series([0,0,0,0,0]) # NOTE: Replace with np.zeros(len(nucleotides_mm))
        if aligned_sequence[index] == original_seq[index]: # If the aligned sequence is same as reference
            pass
        else: # If there is a mismatch, update the encoding vector
            mm_index = nucleotides_mm.index(aligned_sequence[index])
            mm_encoding[mm_index] = 1
        mismatch_mappings_per_position.append(nucleotides_mm)
        encodings_per_position.append(mm_encoding)

    # Create a dataframe from the encodings and the mismatch NT lists
    encodings_per_position_df = pd.DataFrame(encodings_per_position).T
    mismatch_mappings_per_position_df = pd.DataFrame(mismatch_mappings_per_position).T

    encodings_per_position_df.columns = list(original_seq)
    mismatch_mappings_per_position_df.columns = list(original_seq)

    # Prepare the encoding annotated features via a MultiIndex of the position, ref, alt, full_change
    mismatch_mappings_per_position_POS_list = np.arange(mismatch_mappings_per_position_df.shape[1]).repeat(mismatch_mappings_per_position_df.shape[0])
    mismatch_mappings_per_position_REF_list = np.asarray(list(original_seq)).repeat(mismatch_mappings_per_position_df.shape[0]).astype(np.object_)
    mismatch_mappings_per_position_ALT_list = mismatch_mappings_per_position_df.T.values.flatten()
    mismatch_mappings_per_position_full_list = mismatch_mappings_per_position_POS_list.astype(np.str_).astype(object)+mismatch_mappings_per_position_REF_list + np.repeat(">", len(mismatch_mappings_per_position_REF_list)) + mismatch_mappings_per_position_ALT_list
    encodings_per_position_list = encodings_per_position_df.T.values.flatten()
    
    # Encodings per position DF, mismatch mappings per position DF, encodings per position flattened, mismatch mappings per position flattened, mismatch mapping position in flattened list, mismatch mapping ref in flattened list, mismatch mapping alt in flattened list, all substitutions made
    index = pd.MultiIndex.from_tuples(zip(mismatch_mappings_per_position_full_list, mismatch_mappings_per_position_POS_list, mismatch_mappings_per_position_REF_list, mismatch_mappings_per_position_ALT_list), names=["FullChange", "Position","Ref", "Alt"])
    
    assert len(encodings_per_position_list) == len(index)
    encodings_per_position_series = pd.Series(encodings_per_position_list, index = index, name="encoding")
    return encodings_per_position_series


'''
    For a particular row in the CRISPResso table, take the encoding
    
    Removed arguments: encoding
'''
def parse_row(row, original_seq):
    aligned_sequence = row["Aligned_Sequence"]
    reference_sequence = row["Reference_Sequence"]
    insertion_indices = find(reference_sequence, "-")
    aligned_sequence = ''.join([aligned_sequence[i] for i in range(len(aligned_sequence)) if i not in insertion_indices]) # Remove inserted bases from aligned_sequence
    
    assert len(aligned_sequence) == len(original_seq)
    encodings_per_position_series = get_substitution_encoding(aligned_sequence, original_seq)
    return encodings_per_position_series


@dataclass
class EncodingParameters:
    complete_amplicon_sequence: str
    read_length: int
    population_baseline_suffix: Optional[str] = "_baseline"
    population_target_suffix: Optional[str] = "_target"
    population_presort_suffix: Optional[str] = "_presort"
    wt_suffix: Optional[str] = "_wt"

@dataclass
class EncodingDataFrames:
    encoding_parameters: EncodingParameters
    reference_sequence: str
    population_baseline_filepaths: Optional[List[str]] = None
    population_target_filepaths: Optional[List[str]] = None
    population_presort_filepaths: Optional[List[str]] = None
    wt_filepaths: Optional[List[str]] = None
    
    # TODO: Add post check to check if two are assigned, and lengths are the same (except the WT)

    def read_crispresso_allele_tables(self):
        read_allele_table = lambda filename : pd.read_csv(filename, compression='zip', header=0, sep='\t', quotechar='"')
        
        self.population_baseline_df = None if self.population_baseline_filepaths is None else [read_allele_table(fn) for fn in self.population_baseline_filepaths]
        self.population_target_df = None if self.population_target_filepaths is None else [read_allele_table(fn) for fn in self.population_target_filepaths]
        self.population_presort_df = None if self.population_presort_filepaths is None else [read_allele_table(fn) for fn in self.population_presort_filepaths]
        self.population_wt_df = None if self.wt_filepaths is None else [read_allele_table(fn) for fn in self.wt_filepaths]


    def encode_crispresso_allele_table(self, progress_bar=False, cores=1):
        parse_lambda = lambda row: parse_row(row, self.reference_sequence)
        
        if cores > 1:
            pandarallel.initialize(progress_bar=progress_bar, nb_workers=cores)
            print("Encoding population_baseline_df")
            self.population_baseline_encoding = None if self.population_baseline_df is None else [df.parallel_apply(parse_lambda, axis=1) for df in self.population_baseline_df]
            print("Encoding population_target_df")
            self.population_target_encoding = None if self.population_target_df is  None else [df.parallel_apply(parse_lambda, axis=1) for df in self.population_target_df]
            print("Encoding population_presort_df")
            self.population_presort_encoding = None if self.population_presort_df is  None else [df.parallel_apply(parse_lambda, axis=1) for df in self.population_presort_df]
            print("Encoding population_wt_df")
            self.population_wt_encoding = None if self.population_wt_df is  None else [df.parallel_apply(parse_lambda, axis=1) for df in self.population_wt_df]
        else:
            print("Encoding population_baseline_df")
            self.population_baseline_encoding = None if self.population_baseline_df is None else [df.apply(parse_lambda, axis=1) for df in self.population_baseline_df]
            print("Encoding population_target_df")
            self.population_target_encoding = None if self.population_target_df is  None else [df.apply(parse_lambda, axis=1) for df in self.population_target_df]
            print("Encoding population_presort_df")
            self.population_presort_encoding = None if self.population_presort_df is  None else [df.apply(parse_lambda, axis=1) for df in self.population_presort_df]
            print("Encoding population_wt_df")
            self.population_wt_encoding = None if self.population_wt_df is  None else [df.apply(parse_lambda, axis=1) for df in self.population_wt_df]


    def postprocess_encoding(self, guide_edit_positions: List[int] = [], guide_window_halfsize: int = 3, variant_types: List[Tuple[str, str]] = []):
        def trim_edges(trim_left=25, trim_right=25):
            # TODO
            pass
        def process_encoding(encoding_set):
            for encoding_df in encoding_set:
                encoding_df.columns = encoding_df.columns.get_level_values("FullChange")
        def add_read_column(original_dfs, encoded_dfs, suffix):
            for i, original_dfs_rep in enumerate(original_dfs):
                encoded_dfs[i]["#Reads{}".format(suffix)] = original_dfs_rep["#Reads"]

        # Remember to consider strand, spotcheck. Use +6 window size for ABE, +13 window size for evoCDA. +6 window peak
        def denoise_encodings(encoded_dfs, guide_edit_positions: List[int] = [], guide_window_halfsize: int = 3, variant_types: List[Tuple[str, str]] = []):
            
            if (len(guide_edit_positions) > 0) or (len(variant_types) > 0): # If guide positions or variant types are provided, proceed with denoising
                print(f"Denoising with positions {guide_edit_positions} and variant types {variant_types}")
                encoded_dfs_denoised: List[pd.DataFrame] = []

                # For each replicate encoding
                for encoded_df_rep in encoded_dfs:

                    # Get the positions from the column names
                    feature_colnames: List[str] = [name for name in list(encoded_df_rep.columns) if "#Reads" not in name] # List of non-"read" unparsed columns
                    parse_feature = lambda feature : (int(feature[0:feature.index(">")-1]),feature[feature.index(">")-1:feature.index(">")], feature[feature.index(">")+1:], feature)
                    colname_features: List[Tuple[int, str, str, str]] = [parse_feature(feature) for feature in feature_colnames] # List of positions

                    # Get editable positions - we want to remove variants not in these positions
                    editable_positions: List[int] = [editable_position for guide_edit_position in guide_edit_positions for editable_position in range(guide_edit_position-guide_window_halfsize, guide_edit_position+guide_window_halfsize+1)]

                    # Get the features to denoise/remove
                    noneditable_colnames: List[Tuple[int, str, str, str]] = colname_features
                    if editable_positions: # Filter by position
                        noneditable_colnames = [colname_feature for colname_feature in noneditable_colnames if colname_feature[0] not in editable_positions]
                    if len(variant_types) > 0: # Filter by type
                        noneditable_colnames = [colname_feature for colname_feature in noneditable_colnames if np.any([(colname_feature[1]!=variant_type[0]) or (colname_feature[2]!=variant_type[1]) for variant_type in variant_types])]
                    noneditable_features = [colname_tuple[3] for colname_tuple in noneditable_colnames]
                    
                    encoded_df_rep[noneditable_features] = 0
                    encoded_dfs_denoised.append(encoded_df_rep)
                return encoded_dfs_denoised
            else:
                print("Not denoising sample")
                return encoded_dfs
            
        def collapse_encodings(encoded_dfs):
            encoded_dfs_collapsed = []
            for encoded_df_rep in encoded_dfs:
                feature_colnames = [name for name in list(encoded_df_rep.columns) if "#Reads" not in name]
                encoded_dfs_collapsed.append(encoded_df_rep.groupby(feature_colnames, as_index=True).sum().reset_index())
            return encoded_dfs_collapsed
        
        def merge_conditions_by_rep(first_encodings_collapsed, second_encodings_collapsed, third_encodings_collapsed):
            assert len(first_encodings_collapsed) == len(second_encodings_collapsed) == len(third_encodings_collapsed)
            encoded_dfs_merged = []
            for rep_i in range(len(first_encodings_collapsed)):
                feature_colnames = [name for name in list(first_encodings_collapsed[rep_i].columns) if "#Reads" not in name]
                samples = [first_encodings_collapsed[rep_i], second_encodings_collapsed[rep_i], third_encodings_collapsed[rep_i]]
                df_encoding_rep1 = reduce(lambda  left,right: pd.merge(left,right,on=feature_colnames,
                                                        how='outer'), samples).fillna(0)
                
                encoded_dfs_merged.append(df_encoding_rep1)
            return encoded_dfs_merged
        
        # Deep copy encodings
        self.population_baseline_encoding_processed = None if self.population_baseline_encoding is None else copy.deepcopy(self.population_baseline_encoding)
        self.population_target_encoding_processed = None if self.population_target_encoding is None else copy.deepcopy(self.population_target_encoding)
        self.population_presort_encoding_processed = None if self.population_presort_encoding is None else copy.deepcopy(self.population_presort_encoding)
        self.population_wt_encoding_processed = None if self.population_wt_encoding is None else copy.deepcopy(self.population_wt_encoding)

        # Process encodings
        print("Processing encoding columns")
        process_encoding(self.population_baseline_encoding_processed)
        process_encoding(self.population_target_encoding_processed)
        process_encoding(self.population_presort_encoding_processed)
        process_encoding(self.population_wt_encoding_processed)

        # Add read columns to encodings (as a response variable for modelling)
        print("Adding read column")
        add_read_column(self.population_baseline_df, self.population_baseline_encoding_processed, self.encoding_parameters.population_baseline_suffix)
        add_read_column(self.population_target_df, self.population_target_encoding_processed, self.encoding_parameters.population_target_suffix)
        add_read_column(self.population_presort_df, self.population_presort_encoding_processed, self.encoding_parameters.population_presort_suffix)
        add_read_column(self.population_wt_df, self.population_wt_encoding_processed, self.encoding_parameters.wt_suffix)
        
        self.population_baseline_encoding_processed_predenoised = copy.deepcopy(self.population_baseline_encoding_processed)
        self.population_target_encoding_processed_predenoised = copy.deepcopy(self.population_target_encoding_processed)
        self.population_presort_encoding_processed_predenoised = copy.deepcopy(self.population_presort_encoding_processed)
        self.population_wt_encoding_processed_predenoised = copy.deepcopy(self.population_wt_encoding_processed)

        # Denoise encodings
        print("Performing denoising")
        self.population_baseline_encoding_processed = denoise_encodings(self.population_baseline_encoding_processed, guide_edit_positions, guide_window_halfsize, variant_types)
        self.population_target_encoding_processed = denoise_encodings(self.population_target_encoding_processed, guide_edit_positions, guide_window_halfsize, variant_types)
        self.population_presort_encoding_processed = denoise_encodings(self.population_presort_encoding_processed, guide_edit_positions, guide_window_halfsize, variant_types)
        self.population_wt_encoding_processed = denoise_encodings(self.population_wt_encoding_processed, guide_edit_positions, guide_window_halfsize, []) # NOTE: Passing no variant types for WT sample
        
        # Collapse rows with same encodings, sum the reads together.
        print("Collapsing encoding")
        self.population_baseline_encoding_processed = collapse_encodings(self.population_baseline_encoding_processed)
        self.population_target_encoding_processed = collapse_encodings(self.population_target_encoding_processed)
        self.population_presort_encoding_processed = collapse_encodings(self.population_presort_encoding_processed)
        self.population_wt_encoding_processed = collapse_encodings(self.population_wt_encoding_processed)

        self.encodings_collapsed_merged = merge_conditions_by_rep(self.population_baseline_encoding_processed, 
                                                                  self.population_target_encoding_processed, 
                                                                  self.population_presort_encoding_processed)

def save_encodings(encoding_df_list, sort_column, filename):
    for index, encoding_df in enumerate(encoding_df_list):
        encoding_df.sort_values(sort_column, ascending=False).to_csv(filename.format(index), sep="\t")

def save_encodings_df(encoding_df_list, filename):
    for index, encoding_df in enumerate(encoding_df_list):
        encoding_df.to_pickle(filename.format(index))  
