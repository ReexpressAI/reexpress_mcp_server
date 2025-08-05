#########################################################################################################
##################### These are the parameters we used to train the v1.1.0 release.
##################### (Note that the formatting of the output logging and file/directory naming has changed
##################### slightly since this was initially run, but the content is equivalent.)
##################### All of the data is available in the OpenVerification1 dataset on HuggingFace.
##################### The training log is at /model_details/training/logs/run1.log.txt
##################### The eval output is at /model_details/final_eval_output/
#########################################################################################################

cd code/reexpress  # Update with the applicable path

# The conda environment with applicable dependencies is similar to that described in the MCP Server install notes, but uses
#>>> faiss.__version__
#'1.11.0'
#compiled from source for use on H100s. If you want to use faiss-gpu, you also need to change the flag in constants.py to:
#USE_GPU_FAISS_INDEX = True

conda activate model2

RUN_SUFFIX_ID="v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0"
MODEL_TYPE="llm"


DATA_DIR="/home/jupyter/data/processed_reexpress_version1_production_gemini_ibm-granite-3.3-8b-instruct" # Update with the applicable path

TRAIN_FILE="${DATA_DIR}/final/train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/final/calibration.jsonl"

EVAL_LABEL="mmlu_validation"
EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.jsonl"

ALPHA=0.9
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/sagemaker  # Update with the applicable path


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
--epoch 250 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 5 \
--maxQAvailableFromIndexer 2048 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 \
--warm_up_epochs 0 \
--aux_device="cuda:1" \
--pretraining_initialization_epochs=1 \
--pretraining_learning_rate=${LEARNING_RATE} \
--pretraining_initialization_tensors_file="${DATA_DIR}/openthoughts_v4_v5_v6_v7_v8_split1.jsonl.train.jsonl.pretraining.pt" > ${MODEL_OUTPUT_DIR}/run1.log.txt


# Note that in this version, a post-processing step is then needed to create reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db if you want to be able to view the text of the nearest matches from the training set, since the model training script above only saves the training labels, predictions, document_ids, and the exemplar vectors.

# The upper limit of --maxQAvailableFromIndexer 2048 is due to the search limit of this version of FAISS-GPU.

#########################################################################################################
##################### Evaluation
##################### All of the data is available in the OpenVerification1 dataset on HuggingFace.
##################### The eval output is at /model_details/final_eval_output/
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate model2  # conda environment with applicable dependencies (see notes above)

RUN_SUFFIX_ID="v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0"
MODEL_TYPE="llm"


DATA_DIR="/home/jupyter/data/processed_reexpress_version1_production_gemini_ibm-granite-3.3-8b-instruct" # Update with the applicable path

TRAIN_FILE="${DATA_DIR}/final/train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/final/calibration.jsonl"

EVAL_LABEL="mmlu_validation"
EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.jsonl"

ALPHA=0.9
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/sagemaker  # Update with the applicable path


mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00005

###### Comment/uncomment the following, changing the file paths, as applicable. Note that the naming of the files may differ from the fields in the OpenVerification1 dataset on HuggingFace.

#EVAL_LABEL=best_iteration_data_calibration
#EVAL_FILE="/home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/best_iteration_data/calibration.jsonl"  # This is the calibration set of the chosen iteration from the above training run
#
#DATA_DIR="/home/jupyter/data/processed_reexpress_version1_gemini_ibm-granite-3.3-8b-instruct"
#
#EVAL_LABEL="mmlu_validation"
#EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.jsonl"
#
#
DATA_DIR="/home/jupyter/data/processed_reexpress_version1_production_gemini_ibm-granite-3.3-8b-instruct"
EVAL_LABEL="openthoughts_v4_v5_v6_v7_v8_split1.jsonl.test.5k"
EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.jsonl"
#
## 4-question subset of MMLU-Pro. Here, the initial generations are from GPT-4o, over which we verify with this SDM estimator
#EVAL_LABEL="gpt4o_mmlu_pro_test_with_explanations"
#EVAL_LABEL="gpt4o_mmlu_pro_test_only_letters"
#EVAL_FILE="${DATA_DIR}/genai_eval_gpt4o_gemini/${EVAL_LABEL}.jsonl"


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
--epoch 5 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 5 \
--maxQAvailableFromIndexer 2048 \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 \
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--valid_index_conditional_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.valid_index_conditional.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_1.1.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "Possible label errors (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl"
echo "Valid index-conditional predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.valid_index_conditional.jsonl"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_1.1.0.log.txt"


########################################################################
######################## FINAL OUTPUT LOGS /final_eval_output
########################################################################
#Eval Label: best_iteration_data_calibration
#Possible label errors (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.best_iteration_data_calibration.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.best_iteration_data_calibration.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.best_iteration_data_calibration.all_predictions.jsonl
#Eval log file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.best_iteration_data_calibration.version_1.1.0.log.txt
#
#######
#Eval Label: mmlu_validation
#Possible label errors (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.mmlu_validation.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.mmlu_validation.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.mmlu_validation.all_predictions.jsonl
#Eval log file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.mmlu_validation.version_1.1.0.log.txt
#
#######
#Eval Label: gpt4o_mmlu_pro_test_with_explanations
#Possible label errors (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.all_predictions.jsonl
#Eval log file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.version_1.1.0.log.txt
#
#######
#Eval Label: gpt4o_mmlu_pro_test_only_letters
#Possible label errors (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.all_predictions.jsonl
#Eval log file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.version_1.1.0.log.txt
#
#######
#Eval Label: openthoughts_v4_v5_v6_v7_v8_split1.jsonl.test.5k
#Possible label errors (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.openthoughts_v4_v5_v6_v7_v8_split1.jsonl.test.5k.possible_label_errors.jsonl
#Valid index-conditional predictions (sorted) file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.openthoughts_v4_v5_v6_v7_v8_split1.jsonl.test.5k.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.openthoughts_v4_v5_v6_v7_v8_split1.jsonl.test.5k.all_predictions.jsonl
#Eval log file: /home/jupyter/models/v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/sagemaker/final_eval_output/eval.openthoughts_v4_v5_v6_v7_v8_split1.jsonl.test.5k.version_1.1.0.log.txt


#########################################################################################################
##################### Analysis -- Graph ouput -- Note: These graphs are also interactive
#########################################################################################################

########################## Install
#### This expects matplotlib.__version__ == '3.10.0'
#### Install via conda:
#### conda install -c conda-forge matplotlib=3.10.0

########################## Graph and save

cd code/reexpress  # Update with the applicable path

conda activate model2  # conda environment with applicable dependencies (see notes above)

MODEL_OUTPUT_DIR=v2_sdm_estimator_gemini_ibm_granite8b_e250iter5_pretraining_v1_1_0_llm_0.9_1000/ # Update with the applicable path from the github release download
INPUT_DIR="${MODEL_OUTPUT_DIR}/model_details/final_eval_output" # Update with the applicable path


# Choose file to graph/explore
INPUT_FILE=${INPUT_DIR}/eval.best_iteration_data_calibration.all_predictions.jsonl
FILE_LABEL="Calibration (not held-out)"
OUTPUT_FILE_PREFIX="Calibration"
## Note that in v1.1.1, MMLU-validation is held-out
#INPUT_FILE=${INPUT_DIR}/eval.mmlu_validation.all_predictions.jsonl
#FILE_LABEL="MMLU Validation (binary verification)"
#OUTPUT_FILE_PREFIX="MMLU-Validation"
#INPUT_FILE=${INPUT_DIR}/eval.openthoughts_v4_v5_v6_v7_v8_split1.jsonl.test.5k.all_predictions.jsonl
#FILE_LABEL="OpenVerification1 5k Test"
#OUTPUT_FILE_PREFIX="OpenVerification1-5k-Test"
#INPUT_FILE=${INPUT_DIR}/eval.gpt4o_mmlu_pro_test_only_letters.all_predictions.jsonl
#FILE_LABEL="MMLU-Pro-4-QA-GPT4o-Letters"
#OUTPUT_FILE_PREFIX=${FILE_LABEL}
#INPUT_FILE=${INPUT_DIR}/eval.gpt4o_mmlu_pro_test_with_explanations.all_predictions.jsonl
#FILE_LABEL="MMLU-Pro-4-QA-GPT4o-Explanations"
#OUTPUT_FILE_PREFIX=${FILE_LABEL}

OUTPUT_DIR=output_graphs # Update with an applicable path
mkdir ${OUTPUT_DIR}

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_thresholds \
--data_label=${FILE_LABEL} \
--model_version_label="v1.1.1" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_all_points \
--graph_thresholds \
--data_label=${FILE_LABEL} \
--model_version_label="v1.1.1" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

# Add this to make the markers for the wrong predictions larger:
#--emphasize_wrong_predictions


# Pro-tips:
# Click on points to print additional information to the console.
# These plots are zoomable. Click on the magnifying glass in the plot window.
# When --graph_all_points is ommitted, only the valid index-conditional points (i.e., the "admitted"/"non-rejected" points) are displayed. Note that the graphs display the output thresholds by the index of the ground-truth label for reference purposes. Wrong but valid index-conditional points can fall under that horizontal line since the applicable threshold would be for another class. (E.g., you are looking at the graph for label=0, but the model predicted class 1 with a sufficiently high confidence for the point to exceed the threshold for class 1, which could still be lower than the threshold for class 0.) Click on a point for additional details. The thresholds for all classes are printed to console on start and appear in the graph legend.
