#########################################################################################################
##################### Demo data
#########################################################################################################

# The embeddings are from mlx-community/gemma-4-31b-it-4bit, and are constructed from the final-layer
# hidden states:
# max-pool over the sequence :: mean-pool over the sequence :: hidden-state of the final token (that estimates Yes | No)
# Embeddings are 3 x 5376 = 16128 dims.

# This has been tested on an M2 Ultra 76 core 128 GB Mac Studio.

# Substitute your local file paths, where applicable, in the below.


#########################################################################################################
##################### Example of preprocessing the text for generation with another representation model.
##################### Here we keep it simple and only include the existing data for "gpt-5.5-2026-04-23".
#########################################################################################################

conda activate re_mcp_v250  # You also need to install Huggingface datasets (https://huggingface.co/docs/datasets/installation)

cd code/data_processing/code/mlx_examples  # choose applicable path to the repo directory

export HF_HOME=/Users/a/Documents/projects/hf_models/models_cache  # choose applicable HF cache directory for the datasets

DATA_DIR="/Users/a/Documents/projects/data/mcp_server_v2.6.0_gemma"
mkdir -p ${DATA_DIR}

mkdir ${DATA_DIR}/main_data_not_for_public_release

python -u process_hf_v2_4_0_to_jsonl_with_model1_text.py \
--output_train_file="${DATA_DIR}/train.jsonl" \
--output_calibration_file="${DATA_DIR}/calibration.jsonl" \
--output_openthoughts_eval_file="${DATA_DIR}/openthoughts_eval.jsonl" \
--output_mathnet_eval_eval_file="${DATA_DIR}/mathnet_eval.jsonl"

# We use the following for a final run on HLE:
#--input_non_public_hle_eval_file="/Users/a/Documents/projects/data/mcp_server_v2.5.0_new_format/main_data_not_for_public_release/hle_gemini3.1pro_depth0_with_gpt5.5_gemini3.1_verification.jsonl" \
#--output_non_public_hle_eval_file="${DATA_DIR}/main_data_not_for_public_release/hle_gemini3.1pro_depth0_with_gpt5.5_gemini3.1_verification.jsonl"

#Count of rows missing model 1: 1
#Cumulative running time: 347.2261300086975

# neg_1016515_06011677-0c6c-4bc6-a8a2-2fa041b19927 in openthoughts_eval.jsonl is the only blank entry; we'll purposefully keep that as an unusual eval example to examine with the analysis code

#########################################################################################################
##################### Install dependencies
#########################################################################################################

conda create -n re_python_mlx_v1 python=3.12

conda activate re_python_mlx_v1

cd documentation/tutorials/data/code  # choose applicable path to the repo directory

pip install mlx-lm==0.31.3

#########################################################################################################
##################### Process with "mlx-community/gemma-4-31b-it-4bit"
##################### Note that in this case the embedding also includes a max-pool.
#########################################################################################################

conda activate re_python_mlx_v1

cd code/data_processing/code/mlx_examples  # choose applicable path to the repo directory


export HF_HOME=/Users/a/Documents/projects/hf_models/models_cache


DATA_DIR="/Users/a/Documents/projects/data/mcp_server_v2.6.0_gemma"  # choose applicable HF cache directory for the Gemma model (on the order of 20 gb)

MODEL_LABEL="gemma_4_31b_it_4bit"
MAX_TOKEN_LENGTH=8192

# comment out "hle_mc_eval" if just using public data
for INPUT_FILE_NAME in "mathnet_eval" "openthoughts_eval" "calibration" "train"; do # "hle_mc_eval"; do

if [[ "${INPUT_FILE_NAME}" == "hle_mc_eval" ]]; then
    EVAL_FILE="${DATA_DIR}/main_data_not_for_public_release/hle_gemini3.1pro_depth0_with_gpt5.5_gemini3.1_verification.jsonl"
    OUTPUT_DIR="/Users/a/Documents/projects/data/mcp_server_v2.6.0_embedding_${MODEL_LABEL}_maxlength${MAX_TOKEN_LENGTH}/main_data_not_for_public_release/"
    mkdir -p ${OUTPUT_DIR}
else
    EVAL_FILE="${DATA_DIR}/${INPUT_FILE_NAME}.jsonl"
    OUTPUT_DIR="/Users/a/Documents/projects/data/mcp_server_v2.6.0_embedding_${MODEL_LABEL}_maxlength${MAX_TOKEN_LENGTH}"
    mkdir -p ${OUTPUT_DIR}
fi

echo ${INPUT_FILE_NAME}
echo ${OUTPUT_DIR}/${INPUT_FILE_NAME}.logs.txt

python -u add_gemma_4_31b_it_4bit_embeddings_to_v2_4_0_data.py \
--input_file=${EVAL_FILE} \
--class_size=2 \
--max_token_length=${MAX_TOKEN_LENGTH} \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.jsonl >> ${OUTPUT_DIR}/${INPUT_FILE_NAME}.logs.txt 2>&1
done
