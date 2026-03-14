#########################################################################################################
#########################################################################################################
### ALPHA=0.95 # This uses the publicly released model (at 0.9), but we we calibrate at 0.95 on the best
### iteration/epoch of the released model. We then evaluate on the data for which there are
### generations from GPT-5.2 and Gemini-3-pro-preview.
#########################################################################################################
#########################################################################################################

#########################################################################################################
##################### Data
#########################################################################################################

#This assumes you have already prepared the evaluation data as described in
#documentation/model_details/release/v2.1.0_demo_paper/evaluate_released_model_v2.1.0_at_alpha0.9.sh


#########################################################################################################
##################### Download model from GitHub Release v2.1.0 -- note the change of name in the saved directory
##################### We save to a different directory since recalibrating will alter the model files
#########################################################################################################

cd /home/jupyter/models/preview/

wget https://github.com/ReexpressAI/reexpress_mcp_server/releases/download/v2.1.0/reexpress_mcp_server_model__v2_1_0.zip

ALPHA=0.95
mkdir /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_${ALPHA}_1000__recalibrated_alpha

unzip reexpress_mcp_server_model__v2_1_0.zip -d /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_${ALPHA}_1000__recalibrated_alpha/


#########################################################################################################
##################### Recalibrate the v2.1.0 model, which uses alpha=0.9, to alpha=0.95.
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v210

RUN_SUFFIX_ID="sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0"
MODEL_TYPE="release1a"


DATA_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b" # Update with the applicable path

# Recalibration uses the labels and predictions saved with the model. These files are just placeholders.
TRAIN_FILE="${DATA_DIR}/train_filtered/train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/validation/calibration.jsonl"


EVAL_LABEL="mmlu_validation"
EVAL_FILE="${DATA_DIR}/eval/${EVAL_LABEL}.jsonl"

ALPHA=0.95
EXEMPLAR_DIMENSION=1000

UPDATE_SET_LABEL="__recalibrated_alpha"
MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}${UPDATE_SET_LABEL}"/  # Update with the applicable path


mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00005

echo ${MODEL_OUTPUT_DIR}/recalibrate_with_alpha${ALPHA}.log.txt

# Note the use of
#--eval_only \
#--recalibrate_with_updated_alpha


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
--main_device="cuda:0" \
--eval_only \
--recalibrate_with_updated_alpha > ${MODEL_OUTPUT_DIR}/recalibrate_with_alpha${ALPHA}.log.txt


#/home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//recalibrate_with_alpha0.95.log.txt


#########################################################################################################
##################### Evaluation on the GPT-5.2 and Gemini-3 data
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v210

RUN_SUFFIX_ID="sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0"
MODEL_TYPE="release1a"
DATA_DIR="/home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b" # Update with the applicable path
TRAIN_FILE="${DATA_DIR}/train_filtered/train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/validation/calibration.jsonl"
ALPHA=0.95
EXEMPLAR_DIMENSION=1000
UPDATE_SET_LABEL="__recalibrated_alpha"
MODEL_OUTPUT_DIR=/home/jupyter/models/preview/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}${UPDATE_SET_LABEL}"/  # Update with the applicable path
LEARNING_RATE=0.00005


LATEX_MODEL_NAME='modelMergedGPT5.2Gemini3Granite8bTrainingCalibration240kSDMwithRecalibratedAlpha'
MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output_recalibrated
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

#Processing /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/eval/mmlu_validation__only_gpt5.2_and_gemini3.jsonl
#Eval Label: mmlu_validation__only_gpt5.2_and_gemini3
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//final_eval_output_recalibrated/eval.mmlu_validation__only_gpt5.2_and_gemini3.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//final_eval_output_recalibrated/eval.mmlu_validation__only_gpt5.2_and_gemini3.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//final_eval_output_recalibrated/eval.mmlu_validation__only_gpt5.2_and_gemini3.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//final_eval_output_recalibrated/eval.mmlu_validation__only_gpt5.2_and_gemini3.version_2.1.0.log.txt
#
#Processing /home/jupyter/data/openverification1_2026_01_02_updated__preprocessed__granite8b/eval/openthoughts__only_gpt5.2_and_gemini3.jsonl
#Eval Label: openthoughts__only_gpt5.2_and_gemini3
#Possible label errors (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//final_eval_output_recalibrated/eval.openthoughts__only_gpt5.2_and_gemini3.possible_label_errors.jsonl
#High reliablity region predictions (sorted) file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//final_eval_output_recalibrated/eval.openthoughts__only_gpt5.2_and_gemini3.valid_index_conditional.jsonl
#All predictions file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//final_eval_output_recalibrated/eval.openthoughts__only_gpt5.2_and_gemini3.all_predictions.jsonl
#Eval log file: /home/jupyter/models/preview/sdm_verification__merged_gpt5.2_gemini3_granite8b_e500iter5_v2_1_0_release1a_0.95_1000__recalibrated_alpha//final_eval_output_recalibrated/eval.openthoughts__only_gpt5.2_and_gemini3.version_2.1.0.log.txt
