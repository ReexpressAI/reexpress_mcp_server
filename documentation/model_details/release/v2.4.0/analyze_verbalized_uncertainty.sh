#########################################################################################################
##################### Analyze verbalized uncertainty in OpenVerification1.
##################### This will construct the baseline estimators examined in the v2.4.0 model card using the
##################### verbalized uncertainty of GPT-5.5 and Gemini-3.1-pro-preview that appear in
##################### OpenVerification1. The script will print out LaTeX tables, similar to those
##################### generated for the SDM estimator. The code is available in the
##################### `code/data_processing/code/analysis` directory.
#########################################################################################################

conda activate re_mcp_v240  # same environment as version 2.4.0 of the MCP Server. This needs to already exist.
# We also need to install HuggingFace datasets
# We'll make a copy of the environment to avoid any unexpected conflicts with the existing environment for the main code:
conda create --name re_mcp_v240_with_datasets --clone re_mcp_v240

conda activate re_mcp_v240_with_datasets
# install datasets
pip install datasets==4.8.4


cd code/data_processing/code/analysis  # Update with the applicable path

export HF_HOME=/home/jupyter/models/hf

# Directory for the output logs
OUTPUT_DIR=v240_verbalized_uncertainty_analysis/openthoughts_eval  # Update with your chosen path

mkdir -p ${OUTPUT_DIR}

for ALPHA in "0.9" "0.95"; do
echo ${OUTPUT_DIR}/analyze_verbalized_uncertainty_at_${ALPHA}.log.txt
python -u analyze_openverification1_verbalized_uncertainty_gpt5.5_gemini3.1.py \
--alpha=${ALPHA} > ${OUTPUT_DIR}/analyze_verbalized_uncertainty_at_${ALPHA}.log.txt
done


#########################################################################################################
##################### Analyze verbalized uncertainty over OpenVerification1_aux_mathnet
#########################################################################################################

cd code/data_processing/code/analysis  # Update with the applicable path

export HF_HOME=/home/jupyter/models/hf

# Directory for the output logs
OUTPUT_DIR=v240_verbalized_uncertainty_analysis/mathnet_eval  # Update with your chosen path

mkdir -p ${OUTPUT_DIR}

for ALPHA in "0.9" "0.95"; do
echo ${OUTPUT_DIR}/analyze_verbalized_uncertainty_at_${ALPHA}.log.txt
python -u analyze_openverification1_aux_mathnet_verbalized_uncertainty_gpt5.5_gemini3.1.py \
--alpha=${ALPHA} > ${OUTPUT_DIR}/analyze_verbalized_uncertainty_at_${ALPHA}.log.txt
done

