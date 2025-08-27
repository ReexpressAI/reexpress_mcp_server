#########################################################################################################
##################### Preprocess
#########################################################################################################

conda activate re_mcp_v120_gpu  # conda environment with applicable dependencies (see notes in the next section)

cd data_processing/code  # Update with the applicable path

export HF_HOME=hf_models/models_cache  # Update with the applicable path

INPUT_FILE="OpenVerification1"  # This is a locally saved copy of the OpenVerification1 dataset on HuggingFace

OUPUT_DIR="/home/jupyter/data/openverification2__preprocessed_granite8b"

mkdir -p ${OUPUT_DIR}/train
mkdir -p ${OUPUT_DIR}/validation
mkdir -p ${OUPUT_DIR}/eval

python -u prepare_input_format_v1_2_0.py \
--input_datasets_file=${INPUT_FILE} \
--output_train_dir="${OUPUT_DIR}/train" \
--output_validation_dir="${OUPUT_DIR}/validation" \
--output_eval_dir="${OUPUT_DIR}/eval" > "${OUPUT_DIR}/logs.txt"

# We include this for reference on the preprocessing. Note: When run with the current version of the OpenVerification1 dataset on HuggingFace, this will include additional documents not available when this version was trained. Additionally, the distinction of a "pretraining" split is not used here and can be ignored; we only use the instances with GPT-5 (which is used as model1) and Gemini-2.5-pro (which is used as model2).

#########################################################################################################
##################### These are the parameters we used to train the v1.2.0 release.
##################### All of the data is available in the OpenVerification1 dataset on HuggingFace.
##################### In the GitHub Release archive:
##################### The training log is at /model_details/training/logs/run1.log.txt
##################### The eval output is at /model_details/final_eval_output/
#########################################################################################################

cd code/reexpress  # Update with the applicable path

# The conda environment with applicable dependencies is similar to that described in the MCP Server install notes, but uses the following version of FAISS for use on H100s:
#>>> import faiss
#>>> faiss.__version__
#'1.12.0'
# This can be installed from conda via:
# conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.12.0
# Importantly, if you want to use faiss-gpu, you also need to change the flag in constants.py to:
#USE_GPU_FAISS_INDEX = True

conda activate re_mcp_v120_gpu  # conda environment with applicable dependencies (see notes above)

RUN_SUFFIX_ID="sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0"
MODEL_TYPE="r1"


DATA_DIR="/home/jupyter/data/openverification2__preprocessed_granite8b/" # Update with the applicable path (the '2' is an internal label; this is from the OpenVerification1 dataset on HuggingFace; see above)

# The exact document id's used are recorded in model.train_uuids and model.calibration_uuids in the final saved model. (Note that some additional examples using GPT-5 are in OpenVerification1 that were not yet available when this version was trained.)
TRAIN_FILE="${DATA_DIR}/train/train.part1andpart2.jsonl"
CALIBRATION_FILE="${DATA_DIR}/validation/calibration.jsonl"


EVAL_LABEL="mmlu_validation"
EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"

ALPHA=0.9
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path


mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00005

# Note this uses the "embedding" and "attributes" fields in the input JSON lines files because --concat_embeddings_to_attributes is provided as a flag.
python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--concat_embeddings_to_attributes \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--epoch 100 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 3 \
--maxQAvailableFromIndexer 2048 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 \
--warm_up_epochs 0 > ${MODEL_OUTPUT_DIR}/run1.log.txt

# The upper limit of --maxQAvailableFromIndexer 2048 was due to the search limit of earlier versions of FAISS-GPU, which has since been expanded, but we leave unchanged here to avoid compatibility issues.


#########################################################################################################
##################### Construct db: This post-processing step is optional, but creates reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db for viewing the text of the nearest matches from the training set using the MCP Server.
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v120_gpu  # conda environment with applicable dependencies (see notes above)

export HF_HOME=hf_models/models_cache  # Update with the applicable path

python -u aux_construct_document_db.py \
--model_dir="/home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000" \
--best_iteration_train_split_file="/home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000/best_iteration_data/train.jsonl" \
--database_file="/home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000/reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db" \
--input_datasets_file="ReexpressAI/OpenVerification1" \
--dataset_label="OpenVerification1"


#########################################################################################################
##################### Evaluation
##################### All of the data is available in the OpenVerification1 dataset on HuggingFace.
##################### In the GitHub Release archive:
##################### The eval output is at /model_details/final_eval_output/
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v120_gpu  # conda environment with applicable dependencies (see notes above)

RUN_SUFFIX_ID="sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0"
MODEL_TYPE="r1"
DATA_DIR="/home/jupyter/data/openverification2__preprocessed_granite8b/" # Update with the applicable path (the '2' is an internal label; this is from the OpenVerification1 dataset on HuggingFace)
TRAIN_FILE="${DATA_DIR}/train/train.part1andpart2.jsonl"
CALIBRATION_FILE="${DATA_DIR}/validation/calibration.jsonl"
ALPHA=0.9
EXEMPLAR_DIMENSION=1000
MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path
LEARNING_RATE=0.00005

###### Comment/uncomment the following, changing the file paths, as applicable.

#EVAL_LABEL=best_iteration_data_calibration
#EVAL_FILE="/home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000/best_iteration_data/calibration.jsonl"  # This is the calibration set of the chosen iteration from the above training run
#
#EVAL_LABEL="mmlu_validation"
#EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"

EVAL_LABEL="openthoughts"
EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"

## 4-question subset of MMLU-Pro. Here, the initial generations are from GPT-4o, over which we verify with this SDM estimator:

#EVAL_LABEL="gpt4o_mmlu_pro_test_with_explanations"
#EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"
#
#EVAL_LABEL="gpt4o_mmlu_pro_test_only_letters"
#EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"

MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--concat_embeddings_to_attributes \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--epoch 100 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 3 \
--maxQAvailableFromIndexer 2048 \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 \
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--valid_index_conditional_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.valid_index_conditional.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_1.2.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "Possible label errors (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl"
echo "Valid index-conditional predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.valid_index_conditional.jsonl"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_1.2.0.log.txt"


########################################################################
######################## FINAL OUTPUT LOGS /final_eval_output
########################################################################

#Eval Label: best_iteration_data_calibration
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.best_iteration_data_calibration.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.best_iteration_data_calibration.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.best_iteration_data_calibration.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.best_iteration_data_calibration.version_1.2.0.log.txt
#
#Eval Label: mmlu_validation
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.mmlu_validation.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.mmlu_validation.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.mmlu_validation.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.mmlu_validation.version_1.2.0.log.txt
#
#Eval Label: openthoughts
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.openthoughts.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.openthoughts.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.openthoughts.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.openthoughts.version_1.2.0.log.txt
#
#Eval Label: gpt4o_mmlu_pro_test_with_explanations
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.version_1.2.0.log.txt
#
#Eval Label: gpt4o_mmlu_pro_test_only_letters
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.version_1.2.0.log.txt


#########################################################################################################
##################### Analysis -- Graph ouput -- Note: These graphs are also interactive
#########################################################################################################

########################## Graph and save

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v120_gpu  # conda environment with applicable dependencies (see notes above)

MODEL_OUTPUT_DIR=/home/jupyter/models/preview/sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000/ # Update with the applicable path from the github release download
INPUT_DIR="${MODEL_OUTPUT_DIR}/model_details/final_eval_output" # Update with the applicable path


# Choose file to graph/explore
#INPUT_FILE=${INPUT_DIR}/eval.best_iteration_data_calibration.all_predictions.jsonl
#FILE_LABEL="Calibration (not held-out)"
#OUTPUT_FILE_PREFIX="Calibration"
#
## Note that in v1.2.0, MMLU-validation is held-out
#INPUT_FILE=${INPUT_DIR}/eval.mmlu_validation.all_predictions.jsonl
#FILE_LABEL="MMLU Validation (binary verification)"
#OUTPUT_FILE_PREFIX="MMLU-Validation"

INPUT_FILE=${INPUT_DIR}/eval.openthoughts.all_predictions.jsonl
FILE_LABEL="OpenVerification1 5k Test"
OUTPUT_FILE_PREFIX="OpenVerification1-5k-Test"

#INPUT_FILE=${INPUT_DIR}/eval.gpt4o_mmlu_pro_test_only_letters.all_predictions.jsonl
#FILE_LABEL="MMLU-Pro-4-QA-GPT4o-Letters"
#OUTPUT_FILE_PREFIX=${FILE_LABEL}
#
#INPUT_FILE=${INPUT_DIR}/eval.gpt4o_mmlu_pro_test_with_explanations.all_predictions.jsonl
#FILE_LABEL="MMLU-Pro-4-QA-GPT4o-Explanations"
#OUTPUT_FILE_PREFIX=${FILE_LABEL}


OUTPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output/graphs"
mkdir -p ${OUTPUT_DIR}

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_thresholds \
--data_label="${FILE_LABEL}" \
--constant_histogram_count_axis \
--model_version_label="v1.2.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_all_points \
--graph_thresholds \
--data_label="${FILE_LABEL}" \
--constant_histogram_count_axis \
--model_version_label="v1.2.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

# Add this to make the markers for the wrong predictions larger:
#--emphasize_wrong_predictions


# Pro-tips:
# Click on points to print additional information to the console.
# These plots are zoomable. Click on the magnifying glass in the plot window.
# When --graph_all_points is ommitted, only the valid index-conditional points (i.e., the "admitted"/"non-rejected" points) are displayed. Note that the graphs display the output thresholds by the index of the ground-truth label for reference purposes. Wrong but valid index-conditional points can fall under that horizontal line since the applicable threshold would be for another class. (E.g., you are looking at the graph for label=0, but the model predicted class 1 with a sufficiently high confidence for the point to exceed the threshold for class 1, which could still be lower than the threshold for class 0.) Click on a point for additional details. The thresholds for all classes are printed to console on start and appear in the graph legend.

#For reference, if you want to pull up the row in OpenVerification1 for a particular document to retrieve additional information (e.g., output from the older models not currently used in this version), the following can be used:
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
