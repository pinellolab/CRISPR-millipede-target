import pandas as pd
import numpy as np
from pandarallel import pandarallel
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from functools import reduce
import copy
from decimal import Decimal, getcontext, ROUND_HALF_UP

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

''' 
    Function to perform substitution encoding
'''
# DEPRECATED 20251027 replaced for vectorized to improve speed
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

# DEPRECATED 20251027
'''
    For a particular row in the CRISPResso table, take the encoding
    
    Removed arguments: encoding
'''
# DEPRECATED 20251027 replaced for vectorized to improve speed
def parse_row(row, original_seq):
    aligned_sequence = row["Aligned_Sequence"]
    reference_sequence = row["Reference_Sequence"]
    insertion_indices = find(reference_sequence, "-")
    aligned_sequence = ''.join([aligned_sequence[i] for i in range(len(aligned_sequence)) if i not in insertion_indices]) # Remove inserted bases from aligned_sequence
    
    assert len(aligned_sequence) == len(original_seq)
    encodings_per_position_series = get_substitution_encoding(aligned_sequence, original_seq)
    return encodings_per_position_series


# Note 20251027:  from vectorization: https://chatgpt.com/share/68ff9747-34a4-8006-b93f-9be3a4a5c9a6
# Does NOT handle insertions now, since this would prevent equal lengths of each allele thereby preventing straightforward vectorization
import numpy as np
import pandas as pd
import time

def encode_population_df_vectorized(df: pd.DataFrame, reference_seq: str) -> pd.Series:
    """
    Vectorized encoding of a CRISPResso allele table DataFrame.
    - Removes insertions (positions with '-' in Reference_Sequence)
    - Handles substitutions, deletions, and N
    - Returns a pd.Series with MultiIndex (FullChange, Position, Ref, Alt)
      matching the original per-position encoding structure.
    Includes print statements to track progress.
    """
    start_time = time.time()
    print(f"[encode_population_df_vectorized] Started encoding for {len(df)} sequences")

    nucleotides = np.array(["A", "C", "G", "T", "N", "-"])

    ref_list = df["Reference_Sequence"].astype(str).values
    aln_list = df["Aligned_Sequence"].astype(str).values

    print("  - Step 1: Removing insertions from aligned sequences...")
    aln_cleaned = []
    for i, (ref, aln) in enumerate(zip(ref_list, aln_list)):
        if len(ref) != len(aln):
            raise ValueError(f"Row {i}: Reference and aligned sequences differ in length ({len(ref)} vs {len(aln)})")

        aln_clean = ''.join([a for r, a in zip(ref, aln) if r != "-"])
        aln_cleaned.append(aln_clean)

        if (i + 1) % 100000 == 0:
            print(f"    Processed {i + 1:,}/{len(ref_list)} sequences...")

    lengths = {len(s) for s in aln_cleaned}
    if len(lengths) != 1:
        raise ValueError(f"  ❌ Cleaned sequences not uniform in length: {lengths}")

    seq_len = len(aln_cleaned[0])
    if seq_len != len(reference_seq):
        raise ValueError(
            f"  ❌ Cleaned aligned sequence length ({seq_len}) != ungapped reference_seq length ({len(reference_seq)})"
        )
    print(f"  ✓ All sequences cleaned successfully (length = {seq_len})")

    # Step 2: Vectorized conversion
    print("  - Step 2: Converting sequences to numpy arrays...")
    aln_array = np.array([list(seq) for seq in aln_cleaned])
    ref_array = np.array(list(reference_seq))
    print(f"  ✓ Shape of aln_array: {aln_array.shape}")

    # Step 3: Build mismatch mappings
    print("  - Step 3: Building mismatch mappings per position...")
    mismatch_mappings_per_position = []
    for pos in range(seq_len):
        ref_base = ref_array[pos]
        alt_bases = [b for b in nucleotides if b != ref_base]
        mismatch_mappings_per_position.append(alt_bases)

    mismatch_mappings_per_position_df = pd.DataFrame(mismatch_mappings_per_position).T

    # Step 4: Flatten info for MultiIndex
    print("  - Step 4: Creating MultiIndex metadata...")
    mismatch_mappings_per_position_POS_list = np.arange(mismatch_mappings_per_position_df.shape[1]).repeat(
        mismatch_mappings_per_position_df.shape[0]
    )
    mismatch_mappings_per_position_REF_list = np.asarray(list(reference_seq)).repeat(
        mismatch_mappings_per_position_df.shape[0]
    ).astype(object)
    mismatch_mappings_per_position_ALT_list = mismatch_mappings_per_position_df.T.values.flatten()
    mismatch_mappings_per_position_full_list = (
        mismatch_mappings_per_position_POS_list.astype(str)
        + mismatch_mappings_per_position_REF_list
        + ">"
        + mismatch_mappings_per_position_ALT_list
    )

    # Step 5: Compute encodings
    print("  - Step 5: Encoding substitutions and deletions...")
    encodings_per_position = []
    for pos in range(seq_len):
        ref_base = ref_array[pos]
        alt_bases = mismatch_mappings_per_position_df[pos].values
        for alt_base in alt_bases:
            encodings_per_position.append((aln_array[:, pos] == alt_base).astype(int))
        if (pos + 1) % 50 == 0 or pos == seq_len - 1:
            print(f"    Encoded position {pos + 1}/{seq_len}")

    encodings_per_position_array = np.array(encodings_per_position).T  # (n_seq, n_features)
    encoding_df = pd.DataFrame(
        encodings_per_position_array,
        columns=pd.MultiIndex.from_tuples(
            zip(
                mismatch_mappings_per_position_full_list,
                mismatch_mappings_per_position_POS_list,
                mismatch_mappings_per_position_REF_list,
                mismatch_mappings_per_position_ALT_list,
            ),
            names=["FullChange", "Position", "Ref", "Alt"],
        ),
    )
    print(f"  ✓ Encoding complete: shape = {encoding_df.shape} (sequences × features)")
    return encoding_df


@dataclass
class EncodingParameters:
    complete_amplicon_sequence: str
    population_baseline_suffix: Optional[str] = "_baseline"
    population_target_suffix: Optional[str] = "_target"
    population_presort_suffix: Optional[str] = "_presort"
    wt_suffix: Optional[str] = "_wt"
    guide_edit_positions: List[int] = field(default_factory=list)
    guide_window_halfsize: int = 3
    minimum_editing_frequency: float = 0
    minimum_editing_frequency_population: List[str] = field(default_factory=list)
    variant_types: List[Tuple[str, str]] = field(default_factory=list)
    trim_left: int = 0
    trim_right: int = 0
    remove_denoised: bool = False

def sum_technical_replicate_allele_tables(df_list):
    """
    Merge a list of dataframes by the key columns:
    Aligned_Sequence, Reference_Sequence, Reference_Name, Read_Status, n_deleted, n_inserted, n_mutated
    Sum the '#Reads' and recalc '%Reads' as (#Reads / total_reads) * 100 with 16 decimal places.

    Returns a new DataFrame where '%Reads' is a string with 16 decimal places (0..100).
    If you'd prefer Decimal objects instead of strings for '%Reads', see note below.
    """
    # keys to group by
    key_cols = [
        "Aligned_Sequence",
        "Reference_Sequence",
        "Reference_Name",
        "Read_Status",
        "n_deleted",
        "n_inserted",
        "n_mutated",
    ]

    # defensive: ensure input is a list and non-empty
    if not isinstance(df_list, (list, tuple)) or len(df_list) == 0:
        raise ValueError("df_list must be a non-empty list (or tuple) of pandas DataFrames")

    # concat then groupby to perform an outer-merge by keys and sum #Reads
    concat = pd.concat(df_list, ignore_index=True, sort=False)

    # ensure #Reads column exists and is numeric
    if "#Reads" not in concat.columns:
        raise KeyError("#Reads column not found in concatenated dataframe")
    concat["#Reads"] = pd.to_numeric(concat["#Reads"], errors="coerce").fillna(0).astype(int)

    # group and sum #Reads; keep other key columns
    grouped = (
        concat
        .groupby(key_cols, dropna=False, as_index=False)
        .agg({ "#Reads": "sum" })
    )

    # total reads across all grouped rows
    total_reads = int(grouped["#Reads"].sum())

    # use Decimal for high-precision percent calculation
    # set a high precision to avoid rounding issues, then quantize to 16 decimals
    getcontext().prec = 50
    quant = Decimal("0." + ("0" * 15) + "1")   # quantization step for 16 decimal places

    if total_reads == 0:
        # If there are no reads at all, set %Reads to 0.000... (16 decimals)
        grouped["%Reads"] = "0." + ("0" * 16)
        return grouped

    dec_total = Decimal(total_reads)

    def calc_pct_str(nreads_int):
        dec_val = (Decimal(nreads_int) / dec_total) * Decimal(100)
        # round half up to 16 decimal places
        dec_q = dec_val.quantize(quant, rounding=ROUND_HALF_UP)
        # return as string (keeps trailing zeros)
        return format(dec_q, "f")  # or str(dec_q)

    grouped["%Reads"] = grouped["#Reads"].apply(calc_pct_str)

    # Optionally reorder columns: keys, #Reads, %Reads
    out_cols = key_cols + ["#Reads", "%Reads"]
    out_grouped = grouped[out_cols]
    
    # Sort by reads
    out_grouped = out_grouped.sort_values("#Reads", ascending=False)
    return out_grouped

@dataclass
class EncodingDataFrames:
    encoding_parameters: EncodingParameters
    reference_sequence: str
    population_baseline_filepaths: Optional[List[Union[str, List[str]]]] = None
    population_target_filepaths: Optional[List[Union[str, List[str]]]] = None
    population_presort_filepaths: Optional[List[Union[str, List[str]]]] = None
    wt_filepaths: Optional[List[Union[str, List[str]]]] = None
    
    # TODO: Add post check to check if two are assigned, and lengths are the same (except the WT)

    def read_crispresso_allele_tables(self):
        read_allele_table = lambda filename : pd.read_csv(filename, compression='zip', header=0, sep='\t', quotechar='"')
        
        if self.population_baseline_filepaths is None:
            self.population_baseline_df = None
        else:
            population_baseline_df = []
            for biological_replicate_input in self.population_baseline_filepaths:
                if type(biological_replicate_input) is list:
                    biological_replicate_allele_table = sum_technical_replicate_allele_tables([read_allele_table(technical_replicate_fn) for technical_replicate_fn in biological_replicate_input])
                    population_baseline_df.append(biological_replicate_allele_table)
                else:
                    biological_replicate_allele_table = read_allele_table(biological_replicate_input)
                    population_baseline_df.append(biological_replicate_allele_table)
            self.population_baseline_df = population_baseline_df

        if self.population_target_filepaths is None:
            self.population_target_df = None
        else:
            population_target_df = []
            for biological_replicate_input in self.population_target_filepaths:
                if type(biological_replicate_input) is list:
                    biological_replicate_allele_table = sum_technical_replicate_allele_tables([read_allele_table(technical_replicate_fn) for technical_replicate_fn in biological_replicate_input])
                    population_target_df.append(biological_replicate_allele_table)
                else:
                    biological_replicate_allele_table = read_allele_table(biological_replicate_input)
                    population_target_df.append(biological_replicate_allele_table)
            self.population_target_df = population_target_df

        if self.population_presort_filepaths is None:
            self.population_presort_df = None
        else:
            population_presort_df = []
            for biological_replicate_input in self.population_presort_filepaths:
                if type(biological_replicate_input) is list:
                    biological_replicate_allele_table = sum_technical_replicate_allele_tables([read_allele_table(technical_replicate_fn) for technical_replicate_fn in biological_replicate_input])
                    population_presort_df.append(biological_replicate_allele_table)
                else:
                    biological_replicate_allele_table = read_allele_table(biological_replicate_input)
                    population_presort_df.append(biological_replicate_allele_table)
            self.population_presort_df = population_presort_df

        if self.wt_filepaths is None:
            self.population_presort_df = None
        else:
            population_wt_df = []
            for biological_replicate_input in self.wt_filepaths:
                if type(biological_replicate_input) is list:
                    biological_replicate_allele_table = sum_technical_replicate_allele_tables([read_allele_table(technical_replicate_fn) for technical_replicate_fn in biological_replicate_input])
                    population_wt_df.append(biological_replicate_allele_table)
                else:
                    biological_replicate_allele_table = read_allele_table(biological_replicate_input)
                    population_wt_df.append(biological_replicate_allele_table)
            self.population_wt_df = population_wt_df

    def encode_crispresso_allele_table(self):
        """
        Encode all population DataFrames (baseline, target, presort, wt) using
        vectorized substitution encoding. Handles lists of DataFrames.
        """
        # Helper to encode a single population list
        def encode_population_list(pop_list, ref_seq):
            if pop_list is None:
                return None
            # Apply vectorized encoding to each DataFrame in the list
            return [encode_population_df_vectorized(df, ref_seq) for df in pop_list]

        print("Encoding population_baseline_df")
        self.population_baseline_encoding = encode_population_list(self.population_baseline_df, self.reference_sequence)

        print("Encoding population_target_df")
        self.population_target_encoding = encode_population_list(self.population_target_df, self.reference_sequence)

        print("Encoding population_presort_df")
        self.population_presort_encoding = encode_population_list(self.population_presort_df, self.reference_sequence)

        print("Encoding population_wt_df")
        self.population_wt_encoding = encode_population_list(self.population_wt_df, self.reference_sequence)


    def postprocess_encoding(self):
        def trim_edges(encoding_dfs: List[pd.DataFrame], trim_left: int, trim_right: int) -> List[pd.DataFrame]:
            encoded_dfs_trimmed: List[pd.DataFrame] = []
            for encoding_df in encoding_dfs:
                position_indices = encoding_df.columns.get_level_values("Position").astype(int) # Get positions of each column (there will be no read column, this is added in function add_read_column)
                position_left_boundary = min(position_indices) + trim_left
                position_right_boundary = max(position_indices) - trim_right
                encoded_df_trimmed = encoding_df.iloc[:, (position_indices >= position_left_boundary) & (position_indices <= position_right_boundary)]
                encoded_dfs_trimmed.append(encoded_df_trimmed)
            return encoded_dfs_trimmed   
           
        def process_encoding(encoding_set):
            for encoding_df in encoding_set:
                encoding_df.columns = encoding_df.columns.get_level_values("FullChange")
                
        def add_read_column(original_dfs, encoded_dfs, suffix):
            for i, original_dfs_rep in enumerate(original_dfs):
                encoded_dfs[i] = encoded_dfs[i].reset_index(drop=True)
                original_dfs_rep = original_dfs_rep.reset_index(drop=True)
                encoded_dfs[i]["#Reads{}".format(suffix)] = original_dfs_rep["#Reads"]

        # Remember to consider strand, spotcheck. Use +6 window size for ABE, +13 window size for evoCDA. +6 window peak
        def denoise_encodings(encoded_dfs, guide_edit_positions: List[int] = [], guide_window_halfsize: int = 3, variant_types: List[Tuple[str, str]] = [], remove_denoised: bool = False, minimum_editing_frequency: float = 0, minimum_editing_frequency_population: List[str] = []):
            if (len(guide_edit_positions) > 0) or (len(variant_types) > 0): # If guide positions or variant types are provided, proceed with denoising
                print(f"Denoising with positions {guide_edit_positions} and variant types {variant_types}")

                filtered_nucleotide_ids_list: List[str] = []
                # For each replicate encoding, get columns to denoise
                for encoded_df_rep in encoded_dfs:

                    # Get the positions from the column names
                    feature_colnames: List[str] = [name for name in list(encoded_df_rep.columns) if "#Reads" not in name] # List of non-"read" unparsed columns
                    read_colnames: List[str] = [name for name in list(encoded_df_rep.columns) if "#Reads" in name] # List of "read" columns

                    parse_feature = lambda feature : (int(feature[0:feature.index(">")-1]),feature[feature.index(">")-1:feature.index(">")], feature[feature.index(">")+1:], feature)
                    colname_features: List[Tuple[int, str, str, str]] = [parse_feature(feature) for feature in feature_colnames] # List of positions

                    # Get editable positions - we want to remove variants not in these positions
                    editable_positions: List[int] = [editable_position for guide_edit_position in guide_edit_positions for editable_position in range(guide_edit_position-guide_window_halfsize, guide_edit_position+guide_window_halfsize+1)]
                    
                    # Get list of features that are below the minimum editing frequency
                    if (minimum_editing_frequency > 0) and (len(minimum_editing_frequency_population) > 0):
                        print("Filtering by minimum editing frequency")
                        for population in minimum_editing_frequency_population:
                            # Get read column and assert it exists in the dataframe
                            population_read_column = f"#Reads_{population}"
                            assert population_read_column in read_colnames, f"Read column {population_read_column} does not exist do perform minimum editing thresholding, check the parameters of minimum_editing_frequency_population and ensure they are consistent with the population suffixes."
                            
                            # Calculate per-variant allele frequency
                            variant_reads = encoded_df_rep.loc[:, feature_colnames].mul(encoded_df_rep[population_read_column], axis=0).sum(axis=0)
                            variant_af = variant_reads.astype(float).mul(1./encoded_df_rep[population_read_column].sum())
                            variant_af[variant_af.isna()] = 0

                            filtered_nucleotide_ids = variant_af[variant_af<minimum_editing_frequency].index.to_list()
                            filtered_nucleotide_ids_list.extend(filtered_nucleotide_ids)

                    # Get the features to denoise/remove by position and variant type
                    if len(editable_positions) > 0: # Filter by position
                        print("Filtering by editable positions")
                        noneditable_colnames_position = [colname_feature[3] for colname_feature in colname_features if colname_feature[0] not in editable_positions]  # select positions that do not contain position
                        print(f"{len(noneditable_colnames_position)} non-editable positions")
                        filtered_nucleotide_ids_list.extend(noneditable_colnames_position)
                    if len(variant_types) > 0: # Filter by type
                        print("Filtering by variant types")
                        noneditable_colnames_variants = [colname_feature[3] for colname_feature in colname_features if np.all([(colname_feature[1]!=variant_type[0]) or (colname_feature[2]!=variant_type[1]) for variant_type in variant_types])] # Select features that does not contain a variant type
                        print(f"{len(noneditable_colnames_variants)} variant types")
                        filtered_nucleotide_ids_list.extend(noneditable_colnames_variants)
                    
                # Perform denoising
                encoded_dfs_denoised: List[pd.DataFrame] = []
                filtered_nucleotide_ids_set = list(set(filtered_nucleotide_ids_list))
                print(f"Denoising out {len(filtered_nucleotide_ids_set)} columns.")
                if len(filtered_nucleotide_ids_set) > 0:
                    for encoded_df_rep in encoded_dfs:
                        if remove_denoised:
                            encoded_dfs_denoised.append(encoded_df_rep.drop(filtered_nucleotide_ids_set, axis=1))
                        else:
                            encoded_df_rep.loc[:, filtered_nucleotide_ids_set] = 0
                            encoded_dfs_denoised.append(encoded_df_rep)
                    print("Filtered columns, returning denoised DFs")
                    return encoded_dfs_denoised
                print("No columns to filter, returning non-denoised DFs")
                return encoded_dfs
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

        # Trim encodings
        if (self.encoding_parameters.trim_left > 0) or (self.encoding_parameters.trim_right > 0):
            print(f"Trimming encodings with trim_left={self.encoding_parameters.trim_left} and trim_right={self.encoding_parameters.trim_right}")
            self.population_baseline_encoding_processed = trim_edges(self.population_baseline_encoding_processed, self.encoding_parameters.trim_left, self.encoding_parameters.trim_right)
            self.population_target_encoding_processed = trim_edges(self.population_target_encoding_processed, self.encoding_parameters.trim_left, self.encoding_parameters.trim_right)
            self.population_presort_encoding_processed = trim_edges(self.population_presort_encoding_processed, self.encoding_parameters.trim_left, self.encoding_parameters.trim_right)
            self.population_wt_encoding_processed = trim_edges(self.population_wt_encoding_processed, self.encoding_parameters.trim_left, self.encoding_parameters.trim_right)



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
        self.population_baseline_encoding_processed = denoise_encodings(self.population_baseline_encoding_processed, self.encoding_parameters.guide_edit_positions, self.encoding_parameters.guide_window_halfsize, self.encoding_parameters.variant_types, self.encoding_parameters.remove_denoised, self.encoding_parameters.minimum_editing_frequency, self.encoding_parameters.minimum_editing_frequency_population)
        self.population_target_encoding_processed = denoise_encodings(self.population_target_encoding_processed, self.encoding_parameters.guide_edit_positions, self.encoding_parameters.guide_window_halfsize, self.encoding_parameters.variant_types, self.encoding_parameters.remove_denoised, self.encoding_parameters.minimum_editing_frequency, self.encoding_parameters.minimum_editing_frequency_population)
        self.population_presort_encoding_processed = denoise_encodings(self.population_presort_encoding_processed, self.encoding_parameters.guide_edit_positions, self.encoding_parameters.guide_window_halfsize, self.encoding_parameters.variant_types, self.encoding_parameters.remove_denoised, self.encoding_parameters.minimum_editing_frequency, self.encoding_parameters.minimum_editing_frequency_population)
        self.population_wt_encoding_processed = denoise_encodings(self.population_wt_encoding_processed, self.encoding_parameters.guide_edit_positions, self.encoding_parameters.guide_window_halfsize, [], self.encoding_parameters.remove_denoised, self.encoding_parameters.minimum_editing_frequency, self.encoding_parameters.minimum_editing_frequency_population) # NOTE: Passing no variant types for WT sample
        

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
