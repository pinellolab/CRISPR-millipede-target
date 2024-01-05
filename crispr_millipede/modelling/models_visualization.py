# Draft new heatmap with differnet implementation 8/30/2022
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing_extensions import TypeGuard
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

from .models_processing import *
from .models_inputs import *
from typing import Optional

# It seems the best way is an interactive web app for this data - start thinking about how this can be coded in plotly.
class MillipedeModelResultVisualizationExperimentalGroup:
    
    def __init__(self, millipede_model_experimental_group: MillipedeModelExperimentalGroup, 
                 raw_encoding_dfs_experimental_group: RawEncodingDataframesExperimentalGroup, 
                 raw_encodings_editing_freqs_experimental_group: EncodingEditingFrequenciesExperimentalGroup,
                 enriched_pop_label: str,
                 baseline_pop_label: str,
                 presort_pop_label: Optional[str]=None,
                 ctrl_pop_label: Optional[str]=None):
        self.millipede_model_experimental_group = millipede_model_experimental_group
        self.raw_encoding_dfs_experimental_group = raw_encoding_dfs_experimental_group
        self.raw_encodings_editing_freqs_experimental_group = raw_encodings_editing_freqs_experimental_group
        self.enriched_pop_label = enriched_pop_label
        self.baseline_pop_label = baseline_pop_label
        self.presort_pop_label = presort_pop_label
        self.ctrl_pop_label = ctrl_pop_label
        
        #plot_millipede_model_wrapper(millipede_model_experimental_group=millipede_model_experimental_group, 
        #                             raw_encoding_dfs_experimental_group=raw_encoding_dfs_experimental_group, 
        #                             raw_encodings_editing_freqs_experimental_group=raw_encodings_editing_freqs_experimental_group,
        #                             enriched_pop_label=enriched_pop_label,
        #                             baseline_pop_label=baseline_pop_label,
        #                             presort_pop_label=presort_pop_label,
        #                             ctrl_pop_label=ctrl_pop_label)
    
    
    def plot_all_for_model_specification_id(self, 
                                            reference_seq: str,
                                            model_specification_id_list: Optional[List[str]]=None,
                                            model_specification_label_list: Optional[List[str]]=None,
                                            model_types: Optional[List[MillipedeModelType]] = None,
                                            experiment_indices: Optional[List[int]]=None,
                                            replicate_indices: Optional[List[int]]=None,
                                            pdf_filename: Optional[str]= None):
       
        pdf: Optional[PdfPages] = None
        if pdf_filename != None:
            pdf = PdfPages(pdf_filename)
            
        if model_specification_id_list == None:
            model_specification_id_list = self.millipede_model_experimental_group.millipede_model_specification_set_with_results.keys()
        if model_specification_label_list == None:
            model_specification_label_list = model_specification_id_list
          
        assert len(model_specification_label_list) == len(model_specification_id_list), "Model specification label list (len={}) and ID list (len={}) must be same length".format(len(model_specification_label_list), len(model_specification_id_list))
           
        experiment_labels = self.millipede_model_experimental_group.experiments_inputdata.experiment_labels
        reps_labels = self.millipede_model_experimental_group.experiments_inputdata.reps
        for model_specification_id_index, model_specification_id in enumerate(model_specification_id_list):
            model_specification_label = model_specification_label_list[model_specification_id_index]
            
            # Retrieve result(s) for specified specification
             
            millipede_model_specification_result_wrapper: MillipedeModelSpecificationResult= self.millipede_model_experimental_group.millipede_model_specification_set_with_results[model_specification_id]
            millipede_model_specification: MillipedeModelSpecification= millipede_model_specification_result_wrapper.millipede_model_specification
            millipede_model_specification_result_object: Union[MillipedeModelSpecificationSingleMatrixResult, List[MillipedeModelSpecificationSingleMatrixResult], List[List[MillipedeModelSpecificationSingleMatrixResult]]] = millipede_model_specification_result_wrapper.millipede_model_specification_result_input 
            
            # NOTE 03202023: If only a single matrix is provided (i.e. joint model)
            if isinstance(millipede_model_specification_result_object, MillipedeModelSpecificationSingleMatrixResult):
                if experiment_indices != None:
                    print("experiment_indices provided but will not be used as replicates seemed to have been merged")
                if replicate_indices != None:
                    print("replicate_indices provided but will not be used as replicates seemed to have been merged")


                if model_types != None:
                    for model_type in model_types:
                        assert model_type in millipede_model_specification_result_object.millipede_model_specification_single_matrix_result.keys(), "No model for provided model type: {}".format(model_type)
                elif model_types == None:
                    model_types = list(millipede_model_specification_result_object.millipede_model_specification_single_matrix_result.keys())

                for model_type in model_types:
                    print("Showing results for model_type: {}".format(model_type))

                    millipede_model_score_df = millipede_model_specification_result_object.millipede_model_specification_single_matrix_result[model_type].summary.sort_values(by=['PIP'], ascending=False)
                    baseline_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg[0]
                    enriched_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.enriched_pop_encoding_editing_freq_avg[0]
                    presort_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.presort_pop_encoding_editing_freq_avg[0]
                    ctrl_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.ctrl_pop_encoding_editing_freq_avg[0]

                    experiment_label = "+".join(experiment_labels) + " Merge Strategy={}".format(millipede_model_specification.experiment_merge_strategy.value)
                    replicate_label = "+".join(map(str, reps_labels)) + " Merge Strategy={}".format(millipede_model_specification.replicate_merge_strategy.value)
                    model_type_label=model_type.value
                    base_title = "Specification={}; Experiment={}; Replicate={}; Model={}".format(model_specification_label, experiment_label, replicate_label, model_type_label)
                    self.__plot_millipede_model_wrapper(millipede_model_score_df=millipede_model_score_df, 
                                                        original_seq=reference_seq,
                                                        baseline_pop_editing_frequency_avg=baseline_pop_editing_frequency_avg,
                                                        enriched_pop_editing_frequency_avg=enriched_pop_editing_frequency_avg,
                                                        presort_pop_editing_frequency_avg=presort_pop_editing_frequency_avg,
                                                        ctrl_pop_editing_frequency_avg=ctrl_pop_editing_frequency_avg,
                                                        enriched_pop_label=self.enriched_pop_label,
                                                        baseline_pop_label=self.baseline_pop_label,
                                                        presort_pop_label=self.presort_pop_label,
                                                        ctrl_pop_label=self.ctrl_pop_label,
                                                        base_title=base_title,
                                                        pdf=pdf)
            # NOTE 03202023: If only a list of matrices is provided
            elif MillipedeModelResultVisualizationExperimentalGroup.__is_list_of_millipede_model_specification_single_matrix_result(millipede_model_specification_result_object): # -> TypeGuard[List[MillipedeModelSpecificationSingleMatrixResult]]
                millipede_model_specification_result_object: List[MillipedeModelSpecificationSingleMatrixResult]
                if experiment_indices != None:
                    for experimental_index in experiment_indices:
                        assert experimental_index in range(len(millipede_model_specification_result_object)), "Provided experiment_index {} out of range".format(experimental_index) 
                else:
                    experiment_indices = range(len(millipede_model_specification_result_object))

                if replicate_indices != None:
                    print("replicate_index provided but will not be used as replicates seemed to have been merged")

                # Iterate through the experiments in the results object
                for experiment_index in experiment_indices:
                    print("Showing results for experiment index: {}".format(experiment_index))
                    millipede_model_specification_result_object_exp = millipede_model_specification_result_object[experiment_index]

                    # Subset the model types 
                    if model_types != None:
                        for model_type in model_types:
                            # NOTE: Bug in assert statement by comparing object hashes. Instead, compare the enum values
                            # assert model_type in millipede_model_specification_result_object_exp.millipede_model_specification_single_matrix_result.keys(), "No model for provided model type: {}".format(model_type)
                            # TODO: Include the code that actually subsets the model types
                            pass
                    elif model_types == None:
                        model_types = list(millipede_model_specification_result_object_exp.millipede_model_specification_single_matrix_result.keys())



                    for model_type_i, model_type in enumerate(model_types):
                        print("Showing results for model_type {}: {}".format(model_type_i, model_type))
                        millipede_model_score_df = millipede_model_specification_result_object_exp.millipede_model_specification_single_matrix_result[model_type].summary.sort_values(by=['PIP'], ascending=False)
                        print("HERE1")
                        print(len(self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg))
                        baseline_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg[1][experiment_index]
                        enriched_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.enriched_pop_encoding_editing_freq_avg[1][experiment_index]
                        presort_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.presort_pop_encoding_editing_freq_avg[1][experiment_index]
                        ctrl_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.ctrl_pop_encoding_editing_freq_avg[0] # TODO: Hardcoding to the first provided control, but I guess there could be more?

                        experiment_label = experiment_labels[experiment_index]
                        replicate_label = "+".join(map(str, reps_labels)) + " Merge Strategy={}".format(millipede_model_specification.replicate_merge_strategy.value)
                        model_type_label=model_type.value
                        base_title = "Specification={}; Experiment={}; Replicate={}; Model={}".format(model_specification_label, experiment_label, replicate_label, model_type_label)
                        self.__plot_millipede_model_wrapper(millipede_model_score_df=millipede_model_score_df,
                                                            original_seq=reference_seq, 
                                                            baseline_pop_editing_frequency_avg=baseline_pop_editing_frequency_avg,
                                                            enriched_pop_editing_frequency_avg=enriched_pop_editing_frequency_avg,
                                                            presort_pop_editing_frequency_avg=presort_pop_editing_frequency_avg,
                                                            ctrl_pop_editing_frequency_avg=ctrl_pop_editing_frequency_avg,
                                                            enriched_pop_label=self.enriched_pop_label,
                                                            baseline_pop_label=self.baseline_pop_label,
                                                            presort_pop_label=self.presort_pop_label,
                                                            ctrl_pop_label=self.ctrl_pop_label,
                                                            base_title=base_title,
                                                            pdf=pdf)


            elif MillipedeModelResultVisualizationExperimentalGroup.__is_list_of_list_millipede_model_specification_single_matrix_result(millipede_model_specification_result_object): # -> TypeGuard[List[List[MillipedeModelSpecificationSingleMatrixResult]]]
                millipede_model_specification_result_object: List[List[MillipedeModelSpecificationSingleMatrixResult]]

                if experiment_indices != None:
                    for experimental_index in experiment_indices:
                        assert experimental_index in range(len(millipede_model_specification_result_object)), "Provided experiment_index {} out of range".format(experiment_index) 

                else:
                    experiment_indices = range(len(millipede_model_specification_result_object))

                if replicate_indices != None:
                    for millipede_model_specification_result_rep_object in millipede_model_specification_result_object:
                        for replicate_index in replicate_indices:
                            assert replicate_index in range(len(millipede_model_specification_result_rep_object)), "Provided experiment_index {} out of range".format(replicate_index)

                for millipede_model_specification_result_rep_object in millipede_model_specification_result_object:
                    if replicate_indices == None:
                        replicate_indices =  range(len(millipede_model_specification_result_rep_object))

                for experiment_index in experiment_indices:
                    print("Showing results for experiment index: {}".format(experiment_index))
                    millipede_model_specification_result_object_exp = millipede_model_specification_result_object[experiment_index]

                    for replicate_index in replicate_indices:
                        print("Showing results for replicate index: {}".format(replicate_index))
                        millipede_model_specification_result_object_exp_rep = millipede_model_specification_result_object_exp[replicate_index]

                                 # Execute
                        if model_types != None:
                            for model_type in model_types:
                                assert model_type in millipede_model_specification_result_object_exp_rep.millipede_model_specification_single_matrix_result.keys(), "No model for provided model type: {}".format(model_type)
                        elif model_types == None:
                            model_types = list(millipede_model_specification_result_object_exp_rep.millipede_model_specification_single_matrix_result.keys())

                        for model_type in model_types:
                            print("Showing results for model_type: {}".format(model_type))

                            millipede_model_score_df = millipede_model_specification_result_object_exp_rep.millipede_model_specification_single_matrix_result[model_type].summary.sort_values(by=['PIP'], ascending=False)
                            print("HERE2")
                            print(len(self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg))
                            baseline_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.baseline_pop_encoding_editing_freq_avg[1][experiment_index][replicate_index] # NOTE 3/20/2023: Unsure if retrieving replice frequencies appropriately.
                            enriched_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.enriched_pop_encoding_editing_freq_avg[1][experiment_index][replicate_index] # NOTE 3/20/2023: Unsure if retrieving replice frequencies appropriately.
                            presort_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.presort_pop_encoding_editing_freq_avg[1][experiment_index][replicate_index] # NOTE 3/20/2023: Unsure if retrieving replice frequencies appropriately.
                            ctrl_pop_editing_frequency_avg = self.raw_encodings_editing_freqs_experimental_group.ctrl_pop_encoding_editing_freq_avg[0][replicate_index] # TODO: Hardcoding to the first provided control, but I guess there could be more?

                            experiment_label = experiment_labels[experiment_index]
                            replicate_label = reps_labels[replicate_index]
                            model_type_label=model_type.value
                            base_title = "Specification={}; Experiment={}; Replicate={}; Model={}".format(model_specification_label, experiment_label, replicate_label, model_type_label)
                            self.__plot_millipede_model_wrapper(millipede_model_score_df=millipede_model_score_df, 
                                                                baseline_pop_editing_frequency_avg=baseline_pop_editing_frequency_avg,
                                                                enriched_pop_editing_frequency_avg=enriched_pop_editing_frequency_avg,
                                                                presort_pop_editing_frequency_avg=presort_pop_editing_frequency_avg,
                                                                ctrl_pop_editing_frequency_avg=ctrl_pop_editing_frequency_avg,
                                                                enriched_pop_label=self.enriched_pop_label,
                                                                baseline_pop_label=self.baseline_pop_label,
                                                                presort_pop_label=self.presort_pop_label,
                                                                ctrl_pop_label=self.ctrl_pop_label,
                                                                base_title=base_title,
                                                                pdf=pdf)

            else:
                raise Exception("Unexpected type for millipede_model_specification_result_rep_object")

        if pdf != None:
            pdf.close()
           
    @staticmethod
    def __is_list_of_millipede_model_specification_single_matrix_result(val: List[object]) -> TypeGuard[List[MillipedeModelSpecificationSingleMatrixResult]]:
        return all(isinstance(x, MillipedeModelSpecificationSingleMatrixResult) for x in val)
    
    @staticmethod
    def __is_list_of_list_millipede_model_specification_single_matrix_result(val: List[object]) -> TypeGuard[List[List[MillipedeModelSpecificationSingleMatrixResult]]]:
        if all(isinstance(x, list) for x in val) != True:
            return False
        return all(isinstance(y, MillipedeModelSpecificationSingleMatrixResult) for x in val for y in x)

    '''
        Define feature dataframe
    '''
    def __generate_millipede_heatmap_input(self, millipede_model_score_df: pd.DataFrame, original_seq: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nucleotide_ids = [feature for feature in millipede_model_score_df.index if ">" in feature]
        feature_positions = [parse_position(feature) if ">" in feature else None for feature in nucleotide_ids]
        feature_refs = [parse_ref(feature) if ">" in feature else None for feature in nucleotide_ids]
        feature_alts = [parse_alt(feature) if ">" in feature else None for feature in nucleotide_ids]
        feature_meta_df = pd.DataFrame({"positions": feature_positions, "refs": feature_refs, "alts": feature_alts})
        feature_meta_df.index = nucleotide_ids
        feature_meta_df = feature_meta_df.sort_values(["positions", "alts"])
        feature_meta_positions = feature_meta_df[["positions", "refs"]].drop_duplicates()
        
        # Define x-axis and y-axis labels
        xlabels_POS = [format(pos, '03d') + "_" + feature_meta_positions["refs"][i] for i, pos in enumerate(feature_meta_positions["positions"])]
        ylabels_ALT = ["A", "C", "T", "G"]
        
        position_size = len(xlabels_POS)
        nt_size = len(ylabels_ALT)

        # Get heatmap-specific inutsinputs
        coef_position_list = []
        pip_position_list = []
        xlabel_position_list = []
        ylabel_position_list = []

        for position_index, position in enumerate(set(feature_meta_df["positions"])):
            coef_alt_position_array = np.asarray([]) # Model feature coefficient per position and per alt array
            pip_alt_position_array = np.asarray([]) # Model feature PIP per position and per alt array
            xlabel_position_array = np.asarray([]) # Heatmap x-index for each feature
            ylabel_position_array = np.asarray([]) # Heatmap y-index for each feature

            # Get the ref and alt feature for each position
            ref_nt: str = original_seq[position]
            for alt_index, alt_nt in enumerate(ylabels_ALT):
                xlabel_position_array = np.append(xlabel_position_array, xlabels_POS[position]) # Add the X-label, which is from xlabels_POS
                ylabel_position_array = np.append(ylabel_position_array, alt_nt) # Add the y-label, which is the alt_nt


                # If alt and ref are the same, there is no feature for this in the model, so make 0
                if alt_nt == ref_nt:
                    coef_alt_position_array = np.append(coef_alt_position_array, 0)
                    pip_alt_position_array = np.append(pip_alt_position_array, 0)
                else:
                    feature_name = str(position) + ref_nt + ">" + alt_nt  
                    coef_score = millipede_model_score_df.loc[feature_name, "Coefficient"]
                    pip_score = millipede_model_score_df.loc[feature_name, "PIP"]
                    coef_alt_position_array = np.append(coef_alt_position_array, coef_score)
                    pip_alt_position_array = np.append(pip_alt_position_array, pip_score)

            coef_position_list.append(coef_alt_position_array)
            pip_position_list.append(pip_alt_position_array)
            xlabel_position_list.append(xlabel_position_array)
            ylabel_position_list.append(ylabel_position_array)


        coef_vstack = np.vstack(coef_position_list)
        pip_vstack = np.vstack(pip_position_list)
        xlabel_vstack = np.vstack(xlabel_position_list)
        ylabel_vstack = np.vstack(ylabel_position_list)

        coef_flatten = coef_vstack.flatten()
        pip_flatten = pip_vstack.flatten()
        xlabel_flatten = xlabel_vstack.flatten()
        ylabel_flatten = ylabel_vstack.flatten()

        return (coef_flatten, pip_flatten, xlabel_flatten, ylabel_flatten)
   
    
    # Step 1 - Make a scatter plot with square markers, set column names as labels
    def __millipede_results_heatmap(self, x: pd.Series, 
                                    y: pd.Series, 
                                    size: pd.Series, 
                                    color:pd.Series,
                                    alpha: pd.Series, 
                                    editing_freq_groups: List[pd.Series], 
                                    editing_freq_groups_labels: List[str], 
                                    baseline_group_index: int, 
                                    selection_group_index: int, 
                                    overall_group_index: int, 
                                    base_title: Optional[str]=None,
                                    frequency_max: Optional[float]=None, 
                                    frequency_min: Optional[float]=None,
                                    pdf: Optional[PdfPages] = None):
        '''
            3 axes
            axes[0] is the bar plot of per-position mutational frequency
            axes[1] is the dot plot of per-position enrichment
            axes[2] is the heatmap of the Millipede model coefficients 
        '''
        #scale = 0.5# 0.38
        scale=0.25
        #fig, axes = plt.subplots(nrows=3, ncols= 1, figsize=(115*scale,20*scale))
        fig, axes = plt.subplots(nrows=3, ncols= 1, figsize=(30*scale,30*scale)) # TODO ADDED MANUAY 
        axes[0].tick_params(axis='x', which='major', labelsize=8)
        axes[1].tick_params(axis='x', which='major', labelsize=8)
        axes[2].tick_params(axis='x', which='major', labelsize=8)
        fig.suptitle(base_title, fontsize=16)
        fig.tight_layout(pad=2)

        '''
            Preprocess inputs
        '''
        # Mapping from column names to integer coordinates
        x_labels = [v for v in sorted(x.unique())]
        y_labels = [v for v in sorted(y.unique())]
        x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
        y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
        alpha[alpha>1] = 1

        '''
            Configure axes[0]
        '''
        width = 1./(len(editing_freq_groups)+1)
        for index, editing_freq_group in enumerate(editing_freq_groups):
            bar_xval_start = - (width * ((len(editing_freq_groups)+1)/2.))  
            bar_xval_offset = width*(index+1) 
            #print(bar_xval_start)
            axes[0].bar(editing_freq_group.index.values + bar_xval_start + bar_xval_offset, editing_freq_group.values, width, label=editing_freq_groups_labels[index])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axes[0].set_ylabel('Editing Frequency')
        #axes[0].set_xticks([x_to_num[v] for v in x_labels], weight ='bold', labels=[x_to_num[v] for v in x_labels], rotation=45, horizontalalignment='right') TODO MANUALLY REMOVED
        #axes[0].set_xticklabels()
        #axes[1].set_xticklabels(range(len(xlabels)), rotation=45, horizontalalignment='right')
        axes[0].set_xlim(0-1, len(x_labels)+1)
        #axes[0].set_ylim(0, 0.05) # ADDED MANUALLY TODO
        axes[0].set_xlim(200, 250) # ADDED MANUALLY TODO
        #axes[0].grid(which='major') # TODO REMOVED MANUALLY
        axes[0].legend(loc=2)


        '''
            Configure axes[1]
        '''
        enrichment_scores = editing_freq_groups[selection_group_index]/(editing_freq_groups[selection_group_index] + editing_freq_groups[baseline_group_index])
        frequency_scores = editing_freq_groups[overall_group_index]
        cmap = plt.get_cmap("inferno")

        frequency_max_input = np.max(frequency_scores) if frequency_max == None else frequency_max
        frequency_min_input = np.min(frequency_scores) if frequency_min == None else frequency_min

        rescale = lambda y: (y - frequency_min_input) / (frequency_max_input - frequency_min_input)
        rects = axes[1].scatter(range(len(enrichment_scores)), enrichment_scores, color = cmap(rescale(frequency_scores)), s=30)
        axes[1].set_xticks([x_to_num[v] for v in x_labels], weight ='bold', labels=[x_to_num[v] for v in x_labels], rotation=45, horizontalalignment='right')
        #axes[1].set_xticklabels()
        axes[1].set_ylim(-0.1,1.1)
        axes[1].set_xlim(0-1, len(x_labels)+1)
        axes[1].set_xlim(200, 250) # ADDED MANUALLY TODO
        axes[1].axhline(y=0.5, color='black', linestyle='dotted', linewidth=1)
        axes[1].grid(which='major')
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(frequency_min_input,frequency_max_input))
        sm.set_array([])

        cbaxes = inset_axes(axes[1], width="5%", height="5%", loc=2) 
        cbar = plt.colorbar(sm, cax=cbaxes, orientation='horizontal')
        #cbar.set_label('Corrected Mutational Frequency')

        '''
            Configure axes[2]
        '''
        cmap = LinearSegmentedColormap.from_list('tricolor', ['#FF0000', '#FFFFFF', '#0000FF'])

        # Set the range for the color mapping
        vmin = -0.5
        vmax = 0.5

        # Map the values in the z array to colors using the colormap
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        color_values = cmap(norm(color))


        size_scale = 200
        axes[2].scatter(
            x=x.map(x_to_num), # Use mapping for x
            y=y.map(y_to_num), # Use mapping for y
            s=alpha * size_scale, # Vector of square sizes, proportional to size parameter
            color= color_values,
            marker='s' # Use square as scatterplot marker
        )

        # Show column labels on the axes
        axes[2].set_xticks([x_to_num[v] for v in x_labels], weight = 'bold', labels=x_labels, rotation=45, horizontalalignment='right')
        #axes[2].set_xticklabels()
        axes[2].set_xlim(0-1, len(x_labels)+1)
        axes[2].set_xlim(200, 250) # ADDED MANUALLY TODO
        axes[2].set_yticks([y_to_num[v] for v in y_labels])
        axes[2].set_yticklabels(y_labels)
        axes[2].grid(which='major')
        
        if False:# TODO: Temporary removal
            sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin,vmax))
            sm.set_array([])

            cbaxes = inset_axes(axes[2], width="5%", height="5%", loc=2) 
            cbar = plt.colorbar(sm, cax=cbaxes, orientation='horizontal')
            cbar.set_label('Coefficient')
        
        labels = np.asarray([0.1, 0.5, 1.0])
        markersizes = labels*size_scale
        for i in range(len(markersizes)):
            axes[2].scatter([], [], s=markersizes[i], label=labels[i], marker='s', color="blue")

        # Add legend based on the dummy scatter plot
        axes[2].legend(title='PIP', labelspacing=0.7, borderpad=0.3, loc='lower left')


        pdf.savefig(fig)
        plt.show()
    
    
    def __plot_millipede_model_wrapper(self, millipede_model_score_df: pd.DataFrame, 
                                       original_seq:str,
                                       baseline_pop_editing_frequency_avg: pd.Series,
                                       enriched_pop_editing_frequency_avg: pd.Series,
                                       enriched_pop_label: str,
                                       baseline_pop_label: str,
                                       presort_pop_label: Optional[str]=None,
                                       ctrl_pop_label: Optional[str]=None,
                                       base_title: Optional[str]=None,
                                       presort_pop_editing_frequency_avg: Optional[pd.Series] = None,
                                       ctrl_pop_editing_frequency_avg: Optional[pd.Series] = None,
                                       pdf: Optional[PdfPages] = None):
        
        coef_flatten, pip_flatten, xlabel_flatten, ylabel_flatten = self.__generate_millipede_heatmap_input(millipede_model_score_df, original_seq)
        
        print([presort_pop_editing_frequency_avg, enriched_pop_editing_frequency_avg, baseline_pop_editing_frequency_avg, ctrl_pop_editing_frequency_avg])
        self.__millipede_results_heatmap(
            x=pd.Series(xlabel_flatten),
            y=pd.Series(ylabel_flatten),
            size=pd.Series(coef_flatten).abs(),
            color=pd.Series(coef_flatten),
            alpha=pd.Series(pip_flatten).abs(),
            base_title=base_title,
            editing_freq_groups = [presort_pop_editing_frequency_avg, enriched_pop_editing_frequency_avg, baseline_pop_editing_frequency_avg, ctrl_pop_editing_frequency_avg],
            editing_freq_groups_labels = [presort_pop_label, enriched_pop_label, baseline_pop_label, ctrl_pop_label],
            baseline_group_index = 2,
            selection_group_index = 1,
            overall_group_index = 0,
            frequency_min = 0,
            pdf = pdf
        )

        self.__millipede_results_heatmap(
            x=pd.Series(xlabel_flatten),
            y=pd.Series(ylabel_flatten),
            size=pd.Series(coef_flatten).abs(),
            color=pd.Series(coef_flatten),
            alpha=pd.Series(pip_flatten).abs(),
            base_title=base_title,
            editing_freq_groups = [presort_pop_editing_frequency_avg-ctrl_pop_editing_frequency_avg, enriched_pop_editing_frequency_avg-ctrl_pop_editing_frequency_avg, baseline_pop_editing_frequency_avg-ctrl_pop_editing_frequency_avg, ctrl_pop_editing_frequency_avg-ctrl_pop_editing_frequency_avg],
            #editing_freq_groups_labels = [presort_pop_label + "-" + ctrl_pop_label, enriched_pop_label + "-" + ctrl_pop_label, baseline_pop_label + "-" + ctrl_pop_label, ctrl_pop_label + "-" + ctrl_pop_label],
            editing_freq_groups_labels = [presort_pop_label, enriched_pop_label, baseline_pop_label, ctrl_pop_label], # TODO: TEMPORARY for presentation
            baseline_group_index = 2,
            selection_group_index = 1,
            overall_group_index = 0,
            frequency_min = 0,
            pdf = pdf
        )
    
    
#millipede_model_experimental_group: MillipedeModelExperimentalGroup, 
#raw_encoding_dfs_experimental_group: RawEncodingDataframesExperimentalGroup, 
#raw_encodings_editing_freqs_experimental_group: EncodingEditingFrequenciesExperimentalGroup, 
#reference_sequence: str
         