#########################################################################################################
#########################################################################################################
### ALPHA=0.9 # This uses the publicly released model. We evaluate on the data for which there are
### generations from GPT-5.2 and Gemini-3-pro-preview.
#########################################################################################################
#########################################################################################################

#########################################################################################################
##################### Construct embeddings
#########################################################################################################

# First, generate the embeddings for the eval files:
#    INPUT_FILE="mmlu_validation.jsonl"
#    INPUT_FILE="openthoughts.jsonl"
# using add_granite_8b_embeddings_simple.py after running prepare_input_format_v2_1_0.py
# as described at the top of documentation/model_details/release/v2.1.0/train_and_eval_sdm_estimator_v2.1.0.sh

#########################################################################################################
##################### Filter data to only GPT-5.2 and Gemini-3-pro-preview.
#########################################################################################################

cd code/data_processing/code/demo_paper  # Update with the applicable path

conda activate re_mcp_v210  # conda environment for v2.1.0

DATA_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b" # Update with the applicable path from above

for EVAL_LABEL in "mmlu_validation" "openthoughts"; do
python -u filter_data_to_gpt5.2_and_gemini3.py \
--input_file="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl" \
--output_file="${DATA_DIR}/eval/${EVAL_LABEL}__only_gpt5.2_and_gemini3.jsonl"
done

#Number of instances with gpt-5.2-2025-12-11 and gemini-3-pro-preview: 2526
#Total lines: 3036
#
#Number of instances with gpt-5.2-2025-12-11 and gemini-3-pro-preview: 3956
#Total lines: 5000


#########################################################################################################
##################### Download the model from GitHub Release v2.1.0
#########################################################################################################

cd /home/jupyter/models/preview/

wget https://github.com/ReexpressAI/reexpress_mcp_server/releases/download/v2.1.0/reexpress_mcp_server_model__v2_1_0.zip

mkdir /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000
unzip reexpress_mcp_server_model__v2_1_0.zip -d /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000/


#########################################################################################################
##################### Evaluation on the GPT-5.2 and Gemini-3 data
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v210

RUN_SUFFIX_ID="sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0"
MODEL_TYPE="release1a"
DATA_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b"
TRAIN_FILE="${DATA_DIR}/train_filtered/train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/validation/calibration.jsonl"
ALPHA=0.9
EXEMPLAR_DIMENSION=1000
MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path
LEARNING_RATE=0.00005


LATEX_MODEL_NAME='modelMergedGPT5.2Gemini3Granite8bTrainingCalibration240kSDM'
MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}


for EVAL_LABEL in "mmlu_validation__only_gpt5.2_and_gemini3" "openthoughts__only_gpt5.2_and_gemini3"; do
EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"

echo "Processing ${EVAL_FILE}"

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--concat_embeddings_to_attributes \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--eval_batch_size 500 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--main_device="cuda:0" \
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--predictions_in_high_reliability_region_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--construct_results_latex_table_rows \
--additional_latex_meta_data="${EVAL_LABEL},${LATEX_MODEL_NAME}"> ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.1.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "Possible label errors (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl"
echo "High reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.valid_index_conditional.jsonl"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.1.0.log.txt"

done

########################################################################
######################## FINAL OUTPUT LOGS /final_eval_output
########################################################################

#Eval Label: mmlu_validation__only_gpt5.2_and_gemini3
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.mmlu_validation__only_gpt5.2_and_gemini3.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.mmlu_validation__only_gpt5.2_and_gemini3.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.mmlu_validation__only_gpt5.2_and_gemini3.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.mmlu_validation__only_gpt5.2_and_gemini3.version_2.1.0.log.txt
#
#Processing /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/eval/openthoughts__only_gpt5.2_and_gemini3.jsonl
#Eval Label: openthoughts__only_gpt5.2_and_gemini3
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.openthoughts__only_gpt5.2_and_gemini3.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.openthoughts__only_gpt5.2_and_gemini3.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.openthoughts__only_gpt5.2_and_gemini3.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.openthoughts__only_gpt5.2_and_gemini3.version_2.1.0.log.txt


#########################################################################################################
##################### Analysis -- Graph ouput -- Note: These graphs are also interactive
#########################################################################################################

########################## Graph and save

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v210

MODEL_OUTPUT_DIR=/home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000/ # Update with the applicable path from the github release download
INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output" # Update with the applicable path

# Choose file to graph/explore


# Note that in v2.1.0, MMLU-validation is held-out
INPUT_FILE=${INPUT_DIR}/eval.mmlu_validation__only_gpt5.2_and_gemini3.all_predictions.jsonl
FILE_LABEL="MMLU Validation (binary verification)"
OUTPUT_FILE_PREFIX="MMLU-GPT5.2GandGemini3-Validation"
X_AXIS_HISTOGRAM_WIDTH=200

INPUT_FILE=${INPUT_DIR}/eval.openthoughts__only_gpt5.2_and_gemini3.all_predictions.jsonl
FILE_LABEL="OpenVerification1 GPT5.2 and Gemini 3 Test"
OUTPUT_FILE_PREFIX="OpenVerification1-GPT5.2GandGemini3-Test"
X_AXIS_HISTOGRAM_WIDTH=200


OUTPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output/graphs"
mkdir -p ${OUTPUT_DIR}

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_thresholds \
--data_label="${FILE_LABEL}" \
--constant_histogram_count_axis \
--x_axis_histogram_width=${X_AXIS_HISTOGRAM_WIDTH} \
--model_version_label="v2.1.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_all_points \
--graph_thresholds \
--data_label="${FILE_LABEL}" \
--constant_histogram_count_axis \
--x_axis_histogram_width=${X_AXIS_HISTOGRAM_WIDTH} \
--model_version_label="v2.1.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

# Add this to make the markers for the wrong predictions larger:
#--emphasize_wrong_predictions


# Pro-tips:
# Click on points to print additional information to the console.
# These plots are zoomable. Click on the magnifying glass in the plot window.
# When --graph_all_points is ommitted, only the points in the High-Reliability (HR) region (i.e., the "admitted"/"non-rejected" points) are displayed. Note that the graphs display the output thresholds by the index of the ground-truth label for reference purposes. Wrong but admitted points can fall under that horizontal line since the applicable threshold would be for another class. (E.g., you are looking at the graph for label=0, but the model predicted class 1 with a sufficiently high confidence for the point to exceed the threshold for class 1, which could still be lower than the threshold for class 0.) Click on a point for additional details. The thresholds for all classes are printed to console on start and appear in the graph legend.

#For reference, if you want to pull up the row in OpenVerification1 for a particular document to retrieve additional information (e.g., output from the older API-based language models not currently used in this version), the following can be used:
#```python
#from datasets import load_dataset
#dataset = load_dataset("ReexpressAI/OpenVerification1")
#def retrieve_row_by_id(document_id: str):
#    for split_name in ["eval", "validation", "train"]:
#        filtered_dataset = dataset[split_name].filter(lambda x: x['id'] == document_id)
#        if filtered_dataset.num_rows == 1:
#            print(filtered_dataset[0])
#            return filtered_dataset
#```
