#########################################################################################################
##################### Analyze verbalized uncertainty in OpenVerification1.
##################### This will construct the baseline estimators examined in the v2.3.0.preview model card using the
##################### verbalized uncertainty of GPT-5.4 and Gemini-3.1-pro-preview that appear in
##################### OpenVerification1. The script will print out LaTeX tables, similar to those
##################### generated for the SDM estimator. The code is available in the
##################### `code/data_processing/code/analysis` directory.
#########################################################################################################

conda activate re_mcp_v230  # same environment as version 2.3.0 of the MCP Server. This needs to already exist.
# We also need to install HuggingFace datasets
# We'll make a copy of the environment to avoid any unexpected conflicts with the existing environment for the main code:
conda create --name re_mcp_v230_with_datasets --clone re_mcp_v230

conda activate re_mcp_v230_with_datasets
# install datasets
pip install datasets==4.8.4


cd code/data_processing/code/analysis  # Update with the applicable path

export HF_HOME=/home/jupyter/models/hf

# Directory for the output logs
OUTPUT_DIR=v230_verbalized_uncertainty_analysis/  # Update with your chosen path

mkdir ${OUTPUT_DIR}

for ALPHA in "0.9" "0.95"; do
echo ${OUTPUT_DIR}/analyze_verbalized_uncertainty_at_${ALPHA}.log.txt
python -u analyze_openverification1_verbalized_uncertainty_gpt5.4_gemini3.1.py \
--alpha=${ALPHA} > ${OUTPUT_DIR}/analyze_verbalized_uncertainty_at_${ALPHA}.log.txt
done

#analyze_verbalized_uncertainty_at_0.9.log.txt
#
#analyze_verbalized_uncertainty_at_0.95.log.txt
