from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn as sns

# NOTE 20240102: Mostly just copied and pasted the code below, so may need to be modified

def plot_editing_frequency_by_position_histogram(encoding_sample_list, sample_names, title, xlim = None, ylim=0.2):
    number_populations = len(encoding_sample_list)
    number_samples = max([len(encoding_list) for encoding_list in encoding_sample_list])
    fig, axs = plt.subplots(nrows=number_populations, ncols=number_samples, figsize=(number_populations*9, number_samples*5))
    
    if number_populations == 1 and number_samples == 1:
        axs = np.asarray([[axs]])
    
    for population_i, population_encoding_list in enumerate(encoding_sample_list):
        for replicate_j, encoding_df in enumerate(population_encoding_list):
            # This code collapses by position (so dataframe where position is 1 if any of the editing type is TRUE for position)
            encoding_df_position_collapsed = pd.DataFrame([encoding_df.iloc[:, encoding_df.columns.get_level_values("Position") == position].sum(axis=1)>0 for position in encoding_df.columns.levels[1]]).T
            encoding_df_position_collapsed_freq = (encoding_df_position_collapsed.sum(axis=0)/encoding_df_position_collapsed.shape[0])
            if number_populations > 1 or number_samples > 1:
                axs[population_i, replicate_j].bar(range(len(encoding_df_position_collapsed_freq)), encoding_df_position_collapsed_freq)
                axs[population_i, replicate_j].set_xlabel("Position")
            else:
                axs[population_i, replicate_j].bar(range(len(encoding_df_position_collapsed_freq)), encoding_df_position_collapsed_freq)
                axs[population_i, replicate_j].set_xlabel("Position")
            
    for col_i, ax in enumerate(axs[0]):
        ax.set_title("Replicate " + str(col_i + 1))

    for row_i, ax in enumerate(axs[:,0]):
        ax.set_ylabel(sample_names[row_i])
    
    for col_i, _ in enumerate(axs[0]):
        for row_i, _ in enumerate(axs[:,0]):
            axs[col_i, row_i].set_ylim(0, ylim)
            if xlim != None:
                axs[col_i, row_i].set_xlim(xlim[0], xlim[1])
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.show()


def plot_frequency_and_enrichment_by_position(original_seq, editing_frequency_samples, enrichment_samples, baseline_samples, wt_encoding_df, sample_names, title, frequency_max=None, frequency_min=None, position_left=None, position_right=None):
    # Make plot for each replicate - iterate
    fig, axs = plt.subplots(len(editing_frequency_samples), figsize=(12, len(editing_frequency_samples)*5))
    
    
    wt_encoding_df_position_collapsed = pd.DataFrame([wt_encoding_df.iloc[:, wt_encoding_df.columns.get_level_values("Position") == position].sum(axis=1)>0 for position in wt_encoding_df.columns.levels[1]]).T
    wt_encoding_df_position_collapsed_freq = (wt_encoding_df_position_collapsed.sum(axis=0)/wt_encoding_df_position_collapsed.shape[0])
    
    for rep_i in range(len(editing_frequency_samples)):
        frequency_encoding_df = editing_frequency_samples[rep_i]
        enrichment_encoding_df = enrichment_samples[rep_i]
        baseline_encoding_df = baseline_samples[rep_i]
        
        frequency_encoding_df_position_collapsed = pd.DataFrame([frequency_encoding_df.iloc[:, frequency_encoding_df.columns.get_level_values("Position") == position].sum(axis=1)>0 for position in frequency_encoding_df.columns.levels[1]]).T
        enrichment_encoding_df_position_collapsed = pd.DataFrame([enrichment_encoding_df.iloc[:, enrichment_encoding_df.columns.get_level_values("Position") == position].sum(axis=1)>0 for position in enrichment_encoding_df.columns.levels[1]]).T
        baseline_encoding_df_position_collapsed = pd.DataFrame([baseline_encoding_df.iloc[:, baseline_encoding_df.columns.get_level_values("Position") == position].sum(axis=1)>0 for position in baseline_encoding_df.columns.levels[1]]).T
        
        frequency_encoding_df_position_collapsed_freq = (frequency_encoding_df_position_collapsed.sum(axis=0)/frequency_encoding_df_position_collapsed.shape[0])
        enrichment_encoding_df_position_collapsed_freq = (enrichment_encoding_df_position_collapsed.sum(axis=0)/enrichment_encoding_df_position_collapsed.shape[0])
        baseline_encoding_df_position_collapsed_freq = (baseline_encoding_df_position_collapsed.sum(axis=0)/baseline_encoding_df_position_collapsed.shape[0])
        
        enrichment_scores = enrichment_encoding_df_position_collapsed_freq/(enrichment_encoding_df_position_collapsed_freq + baseline_encoding_df_position_collapsed_freq)
        frequency_scores = frequency_encoding_df_position_collapsed_freq - wt_encoding_df_position_collapsed_freq
        cmap = plt.get_cmap("inferno")
        
        frequency_max_input = np.max(frequency_scores) if frequency_max == None else frequency_max
        frequency_min_input = np.min(frequency_scores) if frequency_min == None else frequency_min
        
        rescale = lambda y: (y - frequency_min_input) / (frequency_max_input - frequency_min_input)
        #print(frequency_encoding_df_position_collapsed_freq)
        #print(rescale(frequency_encoding_df_position_collapsed_freq))
        rects = axs[rep_i].scatter(range(len(enrichment_scores)), enrichment_scores, color = cmap(rescale(frequency_scores)), s=15)
        axs[rep_i].set_ylim(0,1)
        axs[rep_i].set_title("Replicate " + str(rep_i))
        axs[rep_i].axhline(y=0.5, color='black', linestyle='dotted', linewidth=1)
        
        if position_left or position_right:
            position_left_xlim = position_left if position_left else 0
            position_right_xlim = position_right if position_right else max(range(len(enrichment_scores)))
            axs[rep_i].set_xlim(position_left_xlim, position_right_xlim)
            for nt in range(position_left_xlim,position_right_xlim+1):
                axs[rep_i].text(nt-0.4, 0.37, original_seq[nt], fontsize=8)
        
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(frequency_min_input,frequency_max_input))
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=axs[rep_i])
        if rep_i == len(editing_frequency_samples)-1:
            cbar.set_label(sample_names[0] + ' mutational frequency \n subtracted by ' + sample_names[3] + ' mutational frequency', rotation=270,labelpad=25)
    plt.xlabel("Amplicon Position")
    plt.ylabel("Editing Frequency of " + sample_names[1] + " over \n summed frequency of " + sample_names[1] + " plus " + sample_names[2])



def plot_trinucleotide_mutational_signature(position_trinucleotide_variants_counts, label, ax):
    #fig, axs = plt.subplots(len(position_trinucleotide_variants_counts), figsize=(10, 30))

    # For normalization and axis setting
    total_mutations = sum([sum(position_trinucleotide_variants_counts_group) for position_trinucleotide_variants_counts_group in position_trinucleotide_variants_counts.values()])
    max_frequency = np.round(max([max(position_trinucleotide_variants_counts_group_series/total_mutations) for position_trinucleotide_variants_counts_group_series in position_trinucleotide_variants_counts.values()]), 2) + 0.01
    
    #for i, (variant, position_trinucleotide_variants_counts_group_series) in enumerate(position_trinucleotide_variants_counts.items()):
    #    rate_percentage = (position_trinucleotide_variants_counts_group_series/total_mutations) * 100
    #    rate_percentage.plot.bar(x='lab', y='val', rot=90, ax=axs[i])
    #    axs[i].set_ylim(0, max_frequency*100)
    #    axs[i].axes.xaxis.set_visible(False)
    #    axs[i].set_title(variant)
    #plt.show()

    '''
        For the annotations
    '''
    current_position = 0
    variant_class_positions = []
    for variant_class, mutsig in position_trinucleotide_variants_counts.items():
        position = current_position + (len(mutsig)/2)
        current_position = current_position + len(mutsig)
        variant_class_positions.append((variant_class, position))
    
    
    color_palette = sns.color_palette("Set2",len(position_trinucleotide_variants_counts))
    bar_colors = np.asarray([color_palette[i] for i, variant_class in enumerate(position_trinucleotide_variants_counts.values()) for _ in range(len(variant_class))])
    pd.concat([(position_trinucleotide_variants_counts_group_series/total_mutations) * 100 for position_trinucleotide_variants_counts_group_series in position_trinucleotide_variants_counts.values()]).plot.bar(x='lab', y='val', rot=0, color = bar_colors, ax=ax)
    ax.set_ylim(0, max_frequency*100)
    ax.set_xticks([position for variant_class, position in variant_class_positions])
    ax.set_xticklabels([variant_class for variant_class, _ in variant_class_positions], rotation=65)
    ax.set_ylabel("Proportion of mutations (%)")
    ax.set_xlabel("Variant Type")
    ax.set_title(label)

from collections import Counter

def plot_mutational_signature(original_seq_input, encoding_df_list, sample_names, title):

    nt_variant_list = ["A","C","G","T"]
    nt_flank_list = ["A","C","G","T","_"]
    position_trinucleotide_variants = {}

    '''
        Get possible trinucleotide context for each variant type
    ''' 
    for nt_ref in nt_variant_list:
        for nt_alt in nt_variant_list:
            if nt_ref != nt_alt:
                position_trinucleotide_variants_group = []
                for nt_left_flank in nt_flank_list:
                    for nt_right_flank in nt_flank_list:
                        position_trinucleotide_variants_group.append((nt_left_flank, nt_ref + ">" + nt_alt,nt_right_flank))
                position_trinucleotide_variants.update({nt_ref + ">" + nt_alt : position_trinucleotide_variants_group})



    fig, axs = plt.subplots(len(encoding_df_list), figsize=(20, 3 * len(encoding_df_list)))
    
    if len(encoding_df_list) == 1:
        axs = [axs]
        
    plt.style.use('seaborn-paper')

    for i, encoding_df in enumerate(encoding_df_list):
        position_trinucleotide_variants_COUNTER = Counter()
        def count_mutations_per_row(read_row):
            mutated_positions = read_row[read_row == 1]
            if len(mutated_positions) > 0:
                pos_all = mutated_positions.index.get_level_values("Position")
                ref_all = mutated_positions.index.get_level_values("Ref")
                alt_all = mutated_positions.index.get_level_values("Alt")
                for i, pos in enumerate(pos_all):
                    pre_nt = None
                    post_nt = None
                    if pos == min(encoding_df.columns.levels[1]):
                        pre_nt = "-"
                        post_nt = original_seq_input[pos+1]
                    elif pos == max(encoding_df.columns.levels[1]):
                        pre_nt = original_seq_input[pos-1]
                        post_nt = "-"
                    else:
                        pre_nt = original_seq_input[pos-1]
                        post_nt = original_seq_input[pos+1]

                    mutation_entry = (pre_nt, ref_all[i] + ">" + alt_all[i], post_nt)
                    position_trinucleotide_variants_COUNTER[mutation_entry] += 1

        _ = encoding_df.loc[encoding_df.sum(axis=1) > 0, :].apply(count_mutations_per_row, axis=1)


        position_trinucleotide_variants_counts = {}
        for position_trinucleotide_variants_group in position_trinucleotide_variants.items():
            position_trinucleotide_variants_counts_group = []
            for trinucleotide_variant in position_trinucleotide_variants_group[1]:
                position_trinucleotide_variants_counts_group.append(position_trinucleotide_variants_COUNTER[trinucleotide_variant])
            position_trinucleotide_variants_counts_group_series = pd.Series(position_trinucleotide_variants_counts_group, index = position_trinucleotide_variants_group[1])
            position_trinucleotide_variants_counts.update({position_trinucleotide_variants_group[0]:position_trinucleotide_variants_counts_group_series})

        plot_trinucleotide_mutational_signature(position_trinucleotide_variants_counts, label = sample_names[i], ax=axs[i])
    fig.suptitle(title, fontsize=16)
    plt.show()

import functools

def plot_editing_frequency_by_position_histogram_variant(encoding_sample_list, sample_names, title, ref_base_list, alt_base_list, xlim = None, ylim=0.2):
    number_populations = len(encoding_sample_list)
    number_samples = max([len(encoding_list) for encoding_list in encoding_sample_list])
    fig, axs = plt.subplots(nrows=number_populations, ncols=number_samples, figsize=(number_populations*9, number_samples*5))
    
    if number_populations == 1 and number_samples == 1:
        axs = np.asarray([[axs]])
    
    for population_i, population_encoding_list in enumerate(encoding_sample_list):
        for replicate_j, encoding_df in enumerate(population_encoding_list):
            # This code collapses by position (so dataframe where position is 1 if any of the editing type is TRUE for position)
            encoding_df_position_collapsed_list = [pd.DataFrame([encoding_df.iloc[:, (encoding_df.columns.get_level_values("Position") == position) & (encoding_df.columns.get_level_values("Ref") == ref_base_list[i]) & (encoding_df.columns.get_level_values("Alt") == alt_base_list[i])].sum(axis=1)>0 for position in encoding_df.columns.levels[1]]).T for i,_ in enumerate(ref_base_list)]
            encoding_df_position_collapsed = functools.reduce(lambda df1, df2: df1 | df2, encoding_df_position_collapsed_list)
            encoding_df_position_collapsed_freq = (encoding_df_position_collapsed.sum(axis=0)/encoding_df_position_collapsed.shape[0])
            if number_populations > 1 or number_samples > 1:
                axs[population_i, replicate_j].bar(range(len(encoding_df_position_collapsed_freq)), encoding_df_position_collapsed_freq)
                axs[population_i, replicate_j].set_xlabel("Position")
            else:
                axs[population_i, replicate_j].bar(range(len(encoding_df_position_collapsed_freq)), encoding_df_position_collapsed_freq)
                axs[population_i, replicate_j].set_xlabel("Position")
            
    for col_i, ax in enumerate(axs[0]):
        ax.set_title("Replicate " + str(col_i + 1))

    for row_i, ax in enumerate(axs[:,0]):
        ax.set_ylabel(sample_names[row_i])
    
    for col_i, _ in enumerate(axs[0]):
        for row_i, _ in enumerate(axs[:,0]):
            axs[col_i, row_i].set_ylim(0, ylim)
            if xlim != None:
                axs[col_i, row_i].set_xlim(xlim[0], xlim[1])
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.show()
