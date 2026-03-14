#########################################################################################################
##################### Analyze verbalized uncertainty in OpenVerification1.
##################### This will construct the estimators examined in the demo paper using the
##################### verbalized uncertainty of GPT-5.2 and Gemini-3-pro-preview that appear in
##################### OpenVerification1. The script will print out LaTeX tables, similar to those
##################### generated for the SDM estimator. The code is available in the
##################### /code/data_processing/code/demo_paper directory.
#########################################################################################################

conda activate re_mcp_v210  # same environment as version 2.1.0 of the MCP Server. This needs to already exist.
# We also need to install HuggingFace datasets
# We'll make a copy of the environment to avoid any unexpected conflicts with the existing environment for the main code:
conda create --name re_mcp_v210_with_datasets --clone re_mcp_v210

conda activate re_mcp_v210_with_datasets
# install datasets
pip install datasets==4.5.0


cd code/data_processing/code/demo_paper  # Update with the applicable path

export HF_HOME=/home/jupyter/models/hf

# Directory for the output logs
OUTPUT_DIR=demo_paper_verbalized_uncertainty_analysis/  # Update with your chosen path

mkdir ${OUTPUT_DIR}

for ALPHA in "0.9" "0.95"; do
echo ${OUTPUT_DIR}/analyze_verbalized_uncertainty_at_${ALPHA}.log.txt
python -u analyze_openverification1_verbalized_uncertainty.py \
--alpha=${ALPHA} > ${OUTPUT_DIR}/analyze_verbalized_uncertainty_at_${ALPHA}.log.txt
done

#analyze_verbalized_uncertainty_at_0.9.log.txt
#
#analyze_verbalized_uncertainty_at_0.95.log.txt
