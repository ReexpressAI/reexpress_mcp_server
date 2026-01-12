#########################################################################################################
##################### Preprocess: Collect the applicable training and evaluation examples from OpenVerification1.
# The v2.1.0 model uses all of the examples in OpenVerification1 for which there is a complete response from:
#    1. GPT-5.2; else GPT-5
#    AND
#    2. Gemini-3-pro-preview; else Gemini-2.5-pro
#########################################################################################################

conda activate re_mcp_v210  # conda environment with applicable dependencies

cd data_processing/code  # Update with the applicable path

export HF_HOME=hf_models/models_cache  # Update with the applicable path

INPUT_FILE="OpenVerification1"  # This is a locally saved copy of the OpenVerification1 dataset on HuggingFace

OUPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed"

mkdir -p ${OUPUT_DIR}/train
mkdir -p ${OUPUT_DIR}/validation
mkdir -p ${OUPUT_DIR}/eval

python -u prepare_input_format_v2_1_0.py \
--input_datasets_file=${INPUT_FILE} \
--output_train_dir="${OUPUT_DIR}/train" \
--output_validation_dir="${OUPUT_DIR}/validation" \
--output_eval_dir="${OUPUT_DIR}/eval" > "${OUPUT_DIR}/logs.txt"

#########################################################################################################
##################### Preprocess/Cache embeddings
#########################################################################################################

# Next, generate the embeddings from the "ibm-granite/granite-3.3-8b-instruct" model and add them
# to the input files. This matches what is used at test time. See the function get_agreement_model_embedding() in
# code/reexpress/mcp_utils_llm_api.py; however, note in the initial preprocessing, we do not apply a
# max character length truncation.

conda activate re_mcp_v210

cd data_processing/code  # Update with the applicable path

export HF_HOME=/home/jupyter/models/hf

for ADDITIONAL_RUN in "1"; do
    DIR_PREFIX="eval"
    INPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed/${DIR_PREFIX}"
    INPUT_FILE="mmlu_validation.jsonl"
    OUTPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/${DIR_PREFIX}"
    mkdir -p ${OUTPUT_DIR}
    
    echo "processing ${INPUT_FILE}"
    
    python -u add_granite_8b_embeddings_simple.py \
    --input_file=${INPUT_DIR}/${INPUT_FILE} \
    --output_file=${OUTPUT_DIR}/${INPUT_FILE}

    for INPUT_FILE in "gpt4o_mmlu_pro_test_only_letters.jsonl" "gpt4o_mmlu_pro_test_with_explanations.jsonl" "openthoughts.jsonl"; do
    echo "processing ${INPUT_FILE}"
    DIR_PREFIX="eval"
    INPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed/${DIR_PREFIX}"
    OUTPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/${DIR_PREFIX}"
    mkdir -p ${OUTPUT_DIR}

    python -u add_granite_8b_embeddings_simple.py \
    --input_file=${INPUT_DIR}/${INPUT_FILE} \
    --output_file=${OUTPUT_DIR}/${INPUT_FILE}
    done

    for INPUT_FILE in "calibration.jsonl"; do
    echo "processing ${INPUT_FILE}"
    DIR_PREFIX="validation"
    INPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed/${DIR_PREFIX}"
    OUTPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/${DIR_PREFIX}"
    mkdir -p ${OUTPUT_DIR}

    python -u add_granite_8b_embeddings_simple.py \
    --input_file=${INPUT_DIR}/${INPUT_FILE} \
    --output_file=${OUTPUT_DIR}/${INPUT_FILE}

    done

    for INPUT_FILE in "train.jsonl"; do
    echo "processing ${INPUT_FILE}"
    DIR_PREFIX="train"
    INPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed/${DIR_PREFIX}"
    OUTPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/${DIR_PREFIX}"
    mkdir -p ${OUTPUT_DIR}

    python -u add_granite_8b_embeddings_simple.py \
    --input_file=${INPUT_DIR}/${INPUT_FILE} \
    --output_file=${OUTPUT_DIR}/${INPUT_FILE}
    done
done


#processing mmlu_validation.jsonl
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 598.4499228000641
#processing gpt4o_mmlu_pro_test_only_letters.jsonl
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 1197.429461479187
#processing gpt4o_mmlu_pro_test_with_explanations.jsonl
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 1234.3636846542358
#processing openthoughts.jsonl
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 1178.0466194152832
#processing calibration.jsonl
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 2488.552804708481
#processing train.jsonl
#Currently processing instance 0
#Currently processing instance 25000
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 1197.429461479187
#processing gpt4o_mmlu_pro_test_with_explanations.jsonl
#Currently processing instance 0ssing instance 150000
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 1234.3636846542358
#processing openthoughts.jsonl
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 1178.0466194152832
#processing calibration.jsonl
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Cumulative running time: 2488.552804708481
#processing train.jsonl
#Currently processing instance 0
#Currently processing instance 25000
#Currently processing instance 50000
#Currently processing instance 75000
#Currently processing instance 100000
#Currently processing instance 125000
#urrently processing instance 150000
#Currently processing instance 175000
#Currently processing instance 200000
#Currently processing instance 225000
#Count of documents with embedding set to 0's: 9
#Cumulative running time: 56219.411435842514

#########################################################################################################
##################### Filter embeddings and collect length summary stats --
# 9 documents in the train split failed to produce embeddings because one of the LM API models went off-the-rails
# with very long repititions of characters. We filter those here, and also collect length statistics.
#########################################################################################################

conda activate re_mcp_v210

cd data_processing/code  # Update with the applicable path

export HF_HOME=/home/jupyter/models/hf


INPUT_FILE="train.jsonl"
DIR_PREFIX="train"
INPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/${DIR_PREFIX}"
OUTPUT_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/${DIR_PREFIX}_filtered"
mkdir -p ${OUTPUT_DIR}

python -u filter_granite_8b_embeddings.py \
--input_file=${INPUT_DIR}/${INPUT_FILE} \
--output_file=${OUTPUT_DIR}/${INPUT_FILE}

#Count of documents with embedding set to 0's: 9
#Among missing embeddings, agreement prompt (in characters): mean: 22902.555555555555, min: 11946, max: 33365
#Among complete embeddings, agreement prompt (in characters): mean: 1415.0271220251577, min: 532, max: 31279
#Cumulative running time: 1245.1724226474762

# Line count:
#229601 /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/train/train.jsonl
#
#229592 /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/train_filtered/train.jsonl

#########################################################################################################
##################### These are the parameters we used to train the v2.1.0 release.
##################### All of the data is available in the OpenVerification1 dataset on HuggingFace.
##################### In the GitHub Release archive:
##################### The training log is at /model_details/training/logs/run1.log.txt
##################### The eval output is at /model_details/final_eval_output/
#########################################################################################################

#cd code/reexpress  # Update with the applicable path

# The conda environment with applicable dependencies is similar to that described in the MCP Server install notes, but uses the following version of FAISS for use on A100s:
#>>> import faiss
#>>> faiss.__version__
#'1.12.0'
# This can be installed from conda via:
# conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.12.0


conda activate re_mcp_v210  # conda environment with applicable dependencies (see notes above)

RUN_SUFFIX_ID="sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0"
MODEL_TYPE="release1a"


DATA_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b" # Update with the applicable path; this is from the OpenVerification1 dataset on HuggingFace; see above)

# The exact document id's used are recorded in model.train_uuids and model.calibration_uuids in the final saved model.
TRAIN_FILE="${DATA_DIR}/train_filtered/train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/validation/calibration.jsonl"


EVAL_LABEL="mmlu_validation"
EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"

ALPHA=0.9
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path


mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00005

echo ${MODEL_OUTPUT_DIR}/run1.log.txt
# Note this uses the "embedding" and "attributes" fields in the input JSON lines files because --concat_embeddings_to_attributes is provided as a flag.
python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--concat_embeddings_to_attributes \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--epoch 500 \
--batch_size 50 \
--eval_batch_size 500 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 5 \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--main_device="cuda:0" > ${MODEL_OUTPUT_DIR}/run1.log.txt

# The upper limit of --maxQAvailableFromIndexer 2048 was due to the search limit of earlier versions of FAISS-GPU, which has since been expanded, but we leave unchanged here to avoid compatibility issues.


#########################################################################################################
##################### Construct db: This post-processing step is optional, but creates reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db for viewing the text of the nearest matches from the training set using the MCP Server.
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v210  # conda environment with applicable dependencies (see notes above)

export HF_HOME=hf_models/models_cache  # Update with the applicable path

RUN_SUFFIX_ID="sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0"
MODEL_TYPE="release1a"
ALPHA=0.9
EXEMPLAR_DIMENSION=1000
# Update paths, as needed:
MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path
mkdir "${MODEL_OUTPUT_DIR}/reexpress_mcp_server_db"

python -u aux_construct_document_db.py \
--model_dir="${MODEL_OUTPUT_DIR}" \
--best_iteration_train_split_file="${MODEL_OUTPUT_DIR}/best_iteration_data/train.jsonl" \
--database_file="${MODEL_OUTPUT_DIR}/reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db" \
--input_datasets_file="ReexpressAI/OpenVerification1" \
--dataset_label="OpenVerification1"

#Successfully loaded ReexpressAI/OpenVerification1, which is the current version from HF Hub.
#Model loaded successfully, set to eval() mode.
#Support set size: 120158
#Database created with 120158 entries.
#Cumulative running time: 968.7106418609619


#########################################################################################################
##################### Evaluation
##################### All of the data is available in the OpenVerification1 dataset on HuggingFace.
##################### In the GitHub Release archive:
##################### The eval output is at /model_details/final_eval_output/
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v210  # conda environment with applicable dependencies (see notes above)

RUN_SUFFIX_ID="sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0"
MODEL_TYPE="release1a"
DATA_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b" # Update with the applicable path; this is from the OpenVerification1 dataset on HuggingFace; see above)
TRAIN_FILE="${DATA_DIR}/train_filtered/train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/validation/calibration.jsonl"
ALPHA=0.9
EXEMPLAR_DIMENSION=1000
MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path
LEARNING_RATE=0.00005


LATEX_MODEL_NAME='modelMergedGPT5.2Gemini3Granite8bTrainingCalibration240kSDM'
MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}


for EVAL_LABEL in "best_iteration_data_calibration" "mmlu_validation" "openthoughts" "gpt4o_mmlu_pro_test_with_explanations" "gpt4o_mmlu_pro_test_only_letters"; do
EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"

# Calibration (i.e., the original input to --input_calibration_set_file) is shuffled during training, so we retrieve the final shuffle associated with this model iteration
if [ "$EVAL_LABEL" = "best_iteration_data_calibration" ]; then
    EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"
fi

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

#Processing /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//best_iteration_data/calibration.jsonl
#
#Eval Label: best_iteration_data_calibration
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.best_iteration_data_calibration.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.best_iteration_data_calibration.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.best_iteration_data_calibration.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.best_iteration_data_calibration.version_2.1.0.log.txt
#
#Processing /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/eval/mmlu_validation.jsonl
#
#Eval Label: mmlu_validation
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.mmlu_validation.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.mmlu_validation.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.mmlu_validation.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.mmlu_validation.version_2.1.0.log.txt
#
#Processing /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/eval/openthoughts.jsonl
#
#Eval Label: openthoughts
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.openthoughts.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.openthoughts.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.openthoughts.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.openthoughts.version_2.1.0.log.txt
#
#Processing /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/eval/gpt4o_mmlu_pro_test_with_explanations.jsonl
#
#Eval Label: gpt4o_mmlu_pro_test_with_explanations
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_with_explanations.version_2.1.0.log.txt
#
#Processing /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/eval/gpt4o_mmlu_pro_test_only_letters.jsonl
#
#Eval Label: gpt4o_mmlu_pro_test_only_letters
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000//final_eval_output/eval.gpt4o_mmlu_pro_test_only_letters.version_2.1.0.log.txt


#########################################################################################################
##################### Analysis -- Graph ouput -- Note: These graphs are also interactive
#########################################################################################################

########################## Graph and save

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v210  # conda environment with applicable dependencies (see notes above)

MODEL_OUTPUT_DIR=/home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.9_1000/ # Update with the applicable path from the github release download
INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output" # Update with the applicable path

# Choose file to graph/explore

INPUT_FILE=${INPUT_DIR}/eval.best_iteration_data_calibration.all_predictions.jsonl
FILE_LABEL="Calibration (not held-out)"
OUTPUT_FILE_PREFIX="Calibration"
X_AXIS_HISTOGRAM_WIDTH=200

# Note that in v2.1.0, MMLU-validation is held-out
INPUT_FILE=${INPUT_DIR}/eval.mmlu_validation.all_predictions.jsonl
FILE_LABEL="MMLU Validation (binary verification)"
OUTPUT_FILE_PREFIX="MMLU-Validation"
X_AXIS_HISTOGRAM_WIDTH=200

INPUT_FILE=${INPUT_DIR}/eval.openthoughts.all_predictions.jsonl
FILE_LABEL="OpenVerification1 5k Test"
OUTPUT_FILE_PREFIX="OpenVerification1-5k-Test"
X_AXIS_HISTOGRAM_WIDTH=200

INPUT_FILE=${INPUT_DIR}/eval.gpt4o_mmlu_pro_test_only_letters.all_predictions.jsonl
FILE_LABEL="MMLU-Pro-4-QA-GPT4o-Letters"
OUTPUT_FILE_PREFIX=${FILE_LABEL}
X_AXIS_HISTOGRAM_WIDTH=200

INPUT_FILE=${INPUT_DIR}/eval.gpt4o_mmlu_pro_test_with_explanations.all_predictions.jsonl
FILE_LABEL="MMLU-Pro-4-QA-GPT4o-Explanations"
OUTPUT_FILE_PREFIX=${FILE_LABEL}
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

