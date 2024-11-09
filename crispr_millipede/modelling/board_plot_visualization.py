import pandas as pd
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

DUMMY_MAPPING_DICT = {"ABE8e":{"A":"A>C","C":"C>A", "G":"G>T", "T":"T>A"}, "evoCDA":{"A":"A>C","C":"C>G", "G":"G>T", "T":"T>A"}}
POSSIBLE_EDITS = {"ABE8e":["A>G", "T>C"], "evoCDA":["C>T", "G>A"]}

def fix_sigma_hits(path, editor, amplicon):
    sigma_hits = pd.read_csv(path)

    for idx, base in enumerate(amplicon):
        new_row = pd.DataFrame([{'Unnamed: 0': f"{idx}{DUMMY_MAPPING_DICT[editor][base]}"}])
        sigma_hits = pd.concat([sigma_hits, new_row], ignore_index=True)

    sigma_hits["sort_column"] = sigma_hits['Unnamed: 0'].str.extract('(^\d+)([A-Z])([->|-])([-]|[A-Z])')[0].astype(float)

    sigma_hits = sigma_hits.sort_values(by="sort_column")
    sigma_hits = sigma_hits.drop(columns=["sort_column"])

    sigma_hits.set_index('Unnamed: 0', inplace=True)
    sigma_hits.index.name = None

    return sigma_hits.fillna(-9999)

def add_dummy_edits(path, editor, amplicon):
    editing = pd.read_csv(path)
    editing = editing[editing["FullChange"].str.contains("|".join(POSSIBLE_EDITS[editor]))]

    for idx, base in enumerate(amplicon):
        for possible_edit in POSSIBLE_EDITS[editor]:
            if f"{idx}{possible_edit}" in editing["FullChange"].values:
                continue
        new_row = pd.DataFrame([{'FullChange': f"{idx}{DUMMY_MAPPING_DICT[editor][base]}", "0":0}])
        editing = pd.concat([editing, new_row], axis=0)

    editing["sort_column"] = editing['FullChange'].str.extract('(^\d+)([A-Z])([->|-])([-]|[A-Z])')[0].astype(float)
    editing = editing.sort_values(by="sort_column")
    editing = editing.drop(columns=["sort_column"])
    editing.columns = ["Edit", "Editing Efficiency"]

    return editing



def millipede_dataframe_cleanup(path, editor, amplicon):

    millipede_dataframe = fix_sigma_hits(path, editor, amplicon)
    millipede_dataframe.reset_index(drop=False, inplace=True)

    #CleanUp dataframe

    millipede_dataframe[['MainIndex','Base','Waste','BaseChange']] = millipede_dataframe['index'].str.extract('(^\d+)([A-Z])([->|-])([-]|[A-Z])')
    millipede_dataframe['BaseChange'] = millipede_dataframe.apply(lambda x: x['Base'] if x['BaseChange']=="-" else x['BaseChange'], axis=1)
    millipede_dataframe['FinalIndex'] = millipede_dataframe['MainIndex'] + millipede_dataframe['Base']
    millipede_dataframe = millipede_dataframe[millipede_dataframe['BaseChange'] != 'N']


    #Generate Betas dataframe
    millipede_dataframe_Betas = millipede_dataframe[['FinalIndex','BaseChange','Coefficient','MainIndex','Base']]
    millipede_dataframe_Betas.loc[:,'MainIndex'] = millipede_dataframe_Betas['MainIndex'].astype(float)
    millipede_dataframe_Betas = millipede_dataframe_Betas.sort_values(by='MainIndex', ascending=True)


    #Generate PIPS dataframe
    millipede_dataframe_PIPS = millipede_dataframe[['FinalIndex','BaseChange','PIP','MainIndex','Base']]
    millipede_dataframe_PIPS.loc[:,'MainIndex'] = millipede_dataframe['MainIndex'].astype(float)
    millipede_dataframe_PIPS = millipede_dataframe_PIPS.sort_values(by='MainIndex', ascending=True)

    #Pivot the Betas dataframe
    
    millipede_dataframe_Betas_pivot = millipede_dataframe_Betas.pivot_table(values='Coefficient', index='BaseChange', columns=['MainIndex','Base'])
    millipede_dataframe_Betas_pivot = millipede_dataframe_Betas_pivot.replace(-9999, np.nan)

    for column in millipede_dataframe_Betas_pivot.columns:
        # Get the Base value for the current column
        base_value = column[1]

        # Set the value at the identified row and column to 0
        millipede_dataframe_Betas_pivot.at[base_value, column] = 0


    #Pivot the PIPS dataframe
    millipede_dataframe_PIPS_pivot = millipede_dataframe_PIPS.pivot_table(values='PIP', index='BaseChange', columns=['MainIndex','Base'])
    millipede_dataframe_PIPS_pivot = millipede_dataframe_PIPS_pivot.replace(-9999, np.nan)

    for column in millipede_dataframe_PIPS_pivot.columns:
        # Get the Base value for the current column
        base_value = column[1]

        # Set the value at the identified row and column to 0
        millipede_dataframe_PIPS_pivot.at[base_value, column] = 0

    return millipede_dataframe_Betas_pivot, millipede_dataframe_PIPS_pivot

def edit_dataframe_cleanup(editor, PresortPath, WTPath, binarized_thresh, amplicon):

    presort = add_dummy_edits(PresortPath, editor, amplicon)
    WT = add_dummy_edits(WTPath, editor, amplicon)

    # Step 2: Rename the 'editing efficiency' columns for clarity
    presort = presort.rename(columns={'Editing Efficiency': 'EditingEfficiencyPresort'})
    WT = WT.rename(columns={'Editing Efficiency': 'EditingEfficiencyWT'})

    # Step 3: Merge the DataFrames on the 'edits' column\
    merged_df = pd.merge(presort[['Edit', 'EditingEfficiencyPresort']], WT[['Edit', 'EditingEfficiencyWT']], on='Edit')
    merged_df['EditingRatio'] = merged_df['EditingEfficiencyPresort']/merged_df['EditingEfficiencyWT']
    merged_df[['MainIndex','Base','Waste','BaseChange']] = merged_df['Edit'].str.extract('(^\d+)([A-Z])([->|-])([-]|[A-Z])')
    # merged_df = merged_df.dropna()
    merged_df['BaseChange'] = merged_df.apply(lambda x: x['Base'] if x['BaseChange']=="-" else x['BaseChange'], axis=1)
    merged_df['FinalIndex'] = merged_df['MainIndex'] + merged_df['Base']
    merged_df = merged_df[merged_df['BaseChange'] != 'N']

    merged_df_ratios = merged_df[['FinalIndex','BaseChange','EditingRatio','MainIndex','Base']]
    merged_df_ratios.loc[:,'MainIndex'] = merged_df_ratios['MainIndex'].astype(float)
    merged_df_ratios = merged_df_ratios.sort_values(by='MainIndex', ascending=True)

    #fill NaN values with -9999
    merged_df_ratios = merged_df_ratios.fillna(-9999)
    merged_df_pivot = merged_df_ratios.pivot_table(values='EditingRatio', index='BaseChange', columns=['MainIndex','Base'])
    merged_df_pivot = merged_df_pivot.replace(-9999, np.nan)

    binarized_df_pivot = merged_df_pivot >= binarized_thresh
    binarized_df_pivot = binarized_df_pivot.astype(float)

    for column in binarized_df_pivot.columns:
      # Get the Base value for the current column
        base_value = column[1]
      # Set the value at the identified row and column to 0
        binarized_df_pivot.at[base_value, column] = 1

        # Initialize an empty list to store the data for the new DataFrame
    df_edit = []

    # Loop through each column of the original DataFrame
    for col in binarized_df_pivot.columns:
        # Get the base level for the column
        base = binarized_df_pivot[col].name[1]

        # Get the values for the column (excluding the 'Base' row)
        values = binarized_df_pivot[col].values

        # Check if more than one row has a value greater than 0
        edit_value = 1 if sum(values > 0) > 1 else 0

        # Append the data to the new_data list
        df_edit.append([base, edit_value])

    # Create the new DataFrame with columns "base" and "edit"
    df_edit = pd.DataFrame(df_edit, columns=['base', 'edit'])

    # Filter the rows with 'edit' value equal to 1
    filtered_rows = df_edit[df_edit['edit'] == 1]

    # Get the index values of the filtered rows and put them into a list
    orange_labels = filtered_rows.index.tolist()

    return binarized_df_pivot, orange_labels


def plot_millipede_boardplot(editor, path, pathPresort, pathWT, start, end, amplicon, binarized_thresh=1.5, fig_width = 8, fig_height = 2, outputPath = None):
    millipede_dataframe_Betas_pivot, millipede_dataframe_PIPS_pivot = millipede_dataframe_cleanup(path, editor, amplicon)
    binarized_df, orange_labels = edit_dataframe_cleanup(editor, pathPresort, pathWT,binarized_thresh, amplicon)

    if millipede_dataframe_Betas_pivot is None or millipede_dataframe_PIPS_pivot is None:
        print("Data not loaded properly")
        return

    millipede_dataframe_Betas_pivot = millipede_dataframe_Betas_pivot.iloc[:, start:end]
    millipede_dataframe_Betas_pivot = millipede_dataframe_Betas_pivot.fillna(0)
    millipede_dataframe_PIPS_pivot = millipede_dataframe_PIPS_pivot.iloc[:, start:end]
    binarized_df = binarized_df.iloc[:, start:end]
    orange_labels = [i - start for i in orange_labels if i >= start and i < end]


    # Dimensions of the figure in inches
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Determine grid dimensions
    grid_rows, grid_cols = millipede_dataframe_Betas_pivot.shape

    # Generate coordinates
    x = np.arange(grid_cols)
    y = np.arange(grid_rows)
    X, Y = np.meshgrid(x, y)

    # Draw grid lines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, grid_cols, 1), minor=False)
    ax.set_yticks(np.arange(-0.5, grid_rows, 1), minor=False)

    # Calculate the size of the axes in points
    fig.canvas.draw()  # This is required to update the figure and axes bounds
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axes_width_pt, axes_height_pt = bbox.width * fig.dpi, bbox.height * fig.dpi

    # Calculate the size of a single grid square in points
    grid_square_width_pt = axes_width_pt / grid_cols
    grid_square_height_pt = axes_height_pt / grid_rows
    grid_square_side_pt = min(grid_square_width_pt, grid_square_height_pt)
    base_size = grid_square_side_pt ** 2


    # Normalize the colors to the range of Betas_pivot values
    norm = plt.Normalize(vmin=-millipede_dataframe_Betas_pivot.values.max(), vmax=millipede_dataframe_Betas_pivot.values.max())

    size_reduction_factor = 0.5

    # Calculate the size of each square in the scatter plot
    size = millipede_dataframe_PIPS_pivot.values * base_size * size_reduction_factor

    # Use scatter plot to color squares
    squares = ax.scatter(X.flatten(), Y.flatten(), s=size.flatten(), c=millipede_dataframe_Betas_pivot.values.flatten(), linewidth=0.1,  cmap='RdBu_r', marker='s', norm=norm, edgecolors='black')

    #Iterate over each square in the binarized DataFrame and add a rectangular patch with a gray facecolor
    for xi, yi, s, val in zip(X.flatten(), Y.flatten(), size.flatten(), binarized_df.values.flatten()):
        if val == 0:
            rect = patches.Rectangle((xi - 0.5, yi - 0.5), 1, 1, linewidth=0, facecolor='gray', alpha=0.5)
            ax.add_patch(rect)

    # Set aspect ratio
    ax.set_aspect('equal')

    # Set the limits of the axes to cover the entire image area
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(-0.5, grid_rows - 0.5)


    # Names for the y-axis labels
    y_labels = ['A', 'C', 'G', 'T']

    ax.set_yticks(np.arange(0.5, grid_rows, 1))
    ax.set_xticks(np.arange(0.5, grid_cols, 1))

    # Set the y-tick labels
    ax.set_yticklabels(y_labels, fontsize = grid_square_width_pt/1.75, verticalalignment='top')

    # Get the tick labels from the second level of the multi-level column index
    tick_labels = millipede_dataframe_Betas_pivot.columns.get_level_values(1)

    # Set the rotation and alignment of the tick labels
    ax.set_xticklabels(tick_labels, rotation=0, ha='right', fontsize = grid_square_width_pt/1.75)

    # Function to add circles to the heatmap cells
    def add_circle(ax, x, y, radius, color):
        circle = plt.Circle((x, y), radius, edgecolor=color, alpha=0.5, linewidth=0.5, fill=False)
        ax.add_patch(circle)

    # Loop through the columns and add circles in the corresponding rows
    for i, column in enumerate(millipede_dataframe_Betas_pivot.columns.get_level_values(1)):
        add_circle(ax, i, millipede_dataframe_Betas_pivot.index.get_loc(column), radius=0.2, color='black')

    for idx, label in enumerate(ax.get_xticklabels()):
        if idx in orange_labels:
            label.set_color('orange')
    # Create a color bar
    cbar = fig.colorbar(squares, ax=ax, orientation = 'horizontal', fraction=0.046)

    # Optionally, add a label to the color bar
    cbar.set_label('Betas', fontsize= grid_square_width_pt/1.5 )
    cbar.ax.tick_params(labelsize=grid_square_width_pt/1.75)
    if outputPath is None:
        base_path = path.replace(".csv", "")  # This removes '.csv' from the end
        svg_path = base_path + "_MYBwPIP" + ".svg"
        fig.savefig(svg_path, format='svg')
    else:
        fig.savefig(outputPath, format='svg')

    plt.show()
