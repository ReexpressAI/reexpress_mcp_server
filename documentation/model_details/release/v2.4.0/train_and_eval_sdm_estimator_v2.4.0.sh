#########################################################################################################
##################### Data
#########################################################################################################

# The full preprocessed input to the SDM estimator is available in the [ReexpressMCPServer_v2_4_0_data](https://huggingface.co/datasets/ReexpressAI/ReexpressMCPServer_v2_4_0_data) dataset.

#########################################################################################################
##################### These are the parameters we used to train the v2.4.0 release.
##################### All of the data is available in the OpenVerification1 dataset on HuggingFace.
##################### In the GitHub Release archive:
##################### The training log is at /model_details/training/logs/run1.log.txt
##################### The eval output is at /model_details/final_eval_output_best_model/
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v240 # conda environment with applicable dependencies

RUN_SUFFIX_ID="reexpress_mcp_server_version_2.4.0_lr0.000001_e700_J10"
MODEL_TYPE="release1a"


DATA_DIR="/home/jupyter/data/v2_4_0/main_data/" # Update with the applicable path

# The training script shuffles D_tr and D_ca. The exact document id's used are recorded in model.train_uuids and model.calibration_uuids in the final saved model.
TRAIN_FILE="${DATA_DIR}/openverification_train_and_adaptation_train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/mathnet_train.jsonl"
EVAL_FILE="${DATA_DIR}/openthoughts_eval.jsonl"

ALPHA=0.9
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path


mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.000001

echo ${MODEL_OUTPUT_DIR}/run1.log.txt
# In this version, we use --use_embeddings, since the "attributes" field has already been concatentated to the "embedding" field in the input JSON lines.
python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--use_embeddings \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--epoch 700 \
--batch_size 50 \
--eval_batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--main_device="cuda:1" > ${MODEL_OUTPUT_DIR}/run1.log.txt

# The upper limit of --maxQAvailableFromIndexer 2048 was due to the search limit of earlier versions of FAISS-GPU, which has since been expanded, but we leave unchanged here to avoid compatibility issues.


#########################################################################################################
##################### Construct db: This post-processing step is optional, but creates reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db for viewing the text of the nearest matches from the training set using the MCP Server.
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v240 # conda environment with applicable dependencies

export HF_HOME=hf_models/models_cache  # Update with the applicable path

# Update paths, as needed:
MODEL_OUTPUT_DIR="/home/jupyter/models/preview/reexpress_mcp_server_version_2.4.0_lr0.000001_e700_J10_release1a_0.9_1000"
mkdir "${MODEL_OUTPUT_DIR}/reexpress_mcp_server_db"

python -u aux_construct_document_db.py \
--model_dir="${MODEL_OUTPUT_DIR}" \
--database_file="${MODEL_OUTPUT_DIR}/reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db" \
--hf_open_verification_datasets_file="ReexpressAI/OpenVerification1" \
--hf_adaptation_datasets_file="ReexpressAI/OpenVerification1_aux_adaptation_examples"

# Use --hf_mathnet_datasets_file="ReexpressAI/OpenVerification1_aux_mathnet" to additionally add the MathNet examples.

#########################################################################################################
##################### Evaluation
##################### All of the data is available in the OpenVerification1 dataset on HuggingFace.
##################### In the GitHub Release archive:
##################### The eval output is at /model_details/final_eval_output_best_model/
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v240 # conda environment with applicable dependencies

RUN_SUFFIX_ID="reexpress_mcp_server_version_2.4.0_lr0.000001_e700_J10"
MODEL_TYPE="release1a"
DATA_DIR="/home/jupyter/data/v2_4_0/main_data/" # Update with the applicable path


TRAIN_FILE="${DATA_DIR}/train_filtered/train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/validation/calibration.jsonl"
ALPHA=0.9
EXEMPLAR_DIMENSION=1000
MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path
LEARNING_RATE=0.000001


LATEX_MODEL_NAME='modelReexpressMCPServerVersion240'
#MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output_best_model
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

STANDARD_OUT_LOG_FILE=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/eval_standard_out.log.txt
echo ${STANDARD_OUT_LOG_FILE}


for EVAL_LABEL in "mathnet_eval" "openthoughts_eval" "best_iteration_data_calibration"; do

EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.jsonl"

# Calibration (i.e., the original input to --input_calibration_set_file) is shuffled during training, so we retrieve the final shuffle associated with this model iteration
if [ "$EVAL_LABEL" = "best_iteration_data_calibration" ]; then
    EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"
fi

echo "Processing ${EVAL_FILE}"

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--use_embeddings \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--eval_batch_size 500 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--main_device="cuda:2" \
--label_error_hr_lower_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.jsonl" \
--predictions_in_high_reliability_region_lower_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.jsonl" \
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--predictions_in_high_reliability_region_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--construct_results_latex_table_rows \
--additional_latex_meta_data="${EVAL_LABEL},${LATEX_MODEL_NAME}"> ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.4.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}" >> ${STANDARD_OUT_LOG_FILE}
echo "Possible label errors in HR_LOWER region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "High reliablity region LOWER predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "Possible label errors in HR region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "High reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.4.0.log.txt" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}

done


#########################################################################################################
##################### Analysis -- Graph ouput -- Note: These graphs are also interactive
#########################################################################################################

########################## Graph and save

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v240 # conda environment with applicable dependencies

MODEL_OUTPUT_DIR=/home/jupyter/models/preview/reexpress_mcp_server_version_2.4.0_lr0.000001_e700_J10_release1a_0.9_1000/ # Update with the applicable path from the github release download

#INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output" # Update with the applicable path
INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output_best_model" # Update with the applicable path

OUTPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output_best_model/graphs"
mkdir -p ${OUTPUT_DIR}

for EVAL_LABEL in "mathnet_eval" "openthoughts_eval" "best_iteration_data_calibration"; do

INPUT_FILE="${INPUT_DIR}/eval.${EVAL_LABEL}.all_predictions.jsonl"

if [ "$EVAL_LABEL" = "best_iteration_data_calibration" ]; then
    FILE_LABEL="Calibration (not held-out)"
    OUTPUT_FILE_PREFIX="Calibration"
    X_AXIS_HISTOGRAM_WIDTH=200
elif [ "$EVAL_LABEL" = "openthoughts_eval" ]; then
    FILE_LABEL="OpenVerification1 5k Test (GPT-5.5, Gemini-3.1)"
    OUTPUT_FILE_PREFIX="OpenVerification1-GPT5_5-Gemini3_1-5k-Test"
    X_AXIS_HISTOGRAM_WIDTH=200
elif [ "$EVAL_LABEL" = "mathnet_eval" ]; then
    FILE_LABEL="MathNet"
    OUTPUT_FILE_PREFIX="MathNet-Eval"
    X_AXIS_HISTOGRAM_WIDTH=200
fi


echo "Processing ${EVAL_FILE}"


python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_thresholds \
--data_label="${FILE_LABEL}" \
--constant_histogram_count_axis \
--x_axis_histogram_width=${X_AXIS_HISTOGRAM_WIDTH} \
--model_version_label="v2.4.0" \
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
--model_version_label="v2.4.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

done

# Note that by default, we now graph sdm(z')_lower on the y-axis. To graph sdm(z'), use --graph_centroid.

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

