
# Installation Instructions for the Reexpress Model-Context-Protocol (MCP) Server
### For tool-calling LLMs (e.g., Claude Opus 4.1 or Sonnet 4) and MCP clients running on Linux or macOS (Sequoia 15 on Apple silicon)

The Reexpress MCP server works with any [MCP client](https://modelcontextprotocol.io/clients). The easiest way to get started is with the [Claude Desktop App](https://claude.ai/download) for macOS Sequoia 15, running on an Apple silicon Mac, since it has web-search (which we highly recommend for verification) built-in as an option. We will assume you have downloaded and installed Claude Desktop in the following.

Installation consists of installing conda, downloading the repo AND model file, installing Python dependencies, and setting file paths. You will also need an [OpenAI API key](https://platform.openai.com/api-keys) or [Microsoft Azure](https://azure.microsoft.com) OpenAI model deployments and corresponding API key (as described below), and a [Google Gemini API key](https://aistudio.google.com/). The IBM Granite model runs locally and will be downloaded from HuggingFace if it is not already in your local model cache directory.

## 1. Install Conda

If you don't have Conda installed, download and install Anaconda Distribution for Apple silicon from the official site:
- [Anaconda Distribution](https://www.anaconda.com/download)

After installation, open your terminal and run:

```bash
echo $(conda info --base)
``` 

This should print something like `/Users/YOUR_USER_NAME/anaconda3`. We will refer to this as your conda path; you'll need it later.

(We use conda rather than uv, which has become common for MCP, since the official release of the [Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) dependency is distributed through conda, as of writing.)

## 2. Clone the Repository

```bash
git clone git clone https://github.com/ReexpressAI/reexpress_mcp_server.git
cd reexpress_mcp_server
pwd
```

That path will be what you'll assign to the variable REEXPRESS_MCP_SERVER_REPO_DIR.

> [!TIP]
> A common source of install errors with MCP servers is not using absolute paths, including for conda. We will use absolute paths in all cases below. For example, if you downloaded the repo to /Users/YOUR_USER_NAME/Documents/, then set REEXPRESS_MCP_SERVER_REPO_DIR=/Users/YOUR_USER_NAME/Documents/reexpress_mcp_server below.

## 3. Download the Model .zip Archive from GitHub Releases

Navigate to the GitHub repository's Releases section and download the required model archive. Unzip the archive to your preferred location. The path to the model directory will be what you'll assign to the variable REEXPRESS_MCP_MODEL_DIR.

## 4. Create and Configure Conda Environment

Run the following lines from the Terminal.

```bash
# Create the Conda environment for the MCP server
conda create -n re_mcp_v120 python=3.12
conda activate re_mcp_v120
pip install torch==2.7.1 transformers==4.53.0 accelerate==1.8.1 numpy==1.26.4
conda install -c pytorch faiss-cpu=1.9.0
pip install "mcp[cli]==1.6.0"
pip install openai==1.70.0
pip install google-genai==1.25.0
conda install -c conda-forge matplotlib=3.10.0
```

This will create an environment "re_mcp_v120" with the required dependencies for macOS 15 and Apple silicon.

## 5. Configure environment variables and LLM API keys

In ${REEXPRESS_MCP_MODEL_DIR} (i.e., the downloaded model directory, NOT the repo directory) is a template file `setup/llm_api_setup.sh` with the following:

```bash
export REEXPRESS_MCP_SERVER_REPO_DIR='/path/to/the/repo_directory'
export REEXPRESS_MCP_MODEL_DIR='/path/to/the/model_directory'

# '0' to use OpenAI models; '1' to use Azure OpenAI deployments
export USE_AZURE_01='0'

### OPENAI_API_KEY is only needed if USE_AZURE_01='0'. Just set to '' or keep this default text if not used
export OPENAI_API_KEY='REPLACE_WITH_API_KEY'

### The following 3 variables are only needed if USE_AZURE_01='1'. Just set to '' or keep this default text if not used
export AZURE_OPENAI_API_KEY='REPLACE_WITH_API_KEY'
export AZURE_OPENAI_ENDPOINT='https://REPLACE_WITH_YOUR_ENDPOINT.azure.com/'
# Fill in with your deployment name for GPT-5. Replace this with whatever name you chose in Azure.
export GPT5_2025_08_07_AZURE_DEPLOYMENT_NAME='gpt-5'

# Google Gemini API key is required
export GEMINI_API_KEY='REPLACE_WITH_API_KEY'

# Optionally, set a cache directory for "ibm-granite/granite-3.3-8b-instruct" by uncommenting the following and adding a valid path
# export HF_HOME=''

# '1' to create the HTML page in ${REEXPRESS_MCP_MODEL_DIR}/visualize/current_reexpression.html for each call to the MCP server; '0' turns this feature off
export REEXPRESS_MCP_SAVE_OUTPUT='0'
```

Replace REEXPRESS_MCP_SERVER_REPO_DIR and REEXPRESS_MCP_MODEL_DIR with the repo and model directory from above, respectively. If you want to use OpenAI, set USE_AZURE_01='0' and supply your API key in OPENAI_API_KEY. If instead, you want to use Azure, set USE_AZURE_01='1' and fill in the corresponding 3 variables. Optionally add a Huggingface model cache directory. Finally, set REEXPRESS_MCP_SAVE_OUTPUT='1' if you want to create an HTML page for the tool call output.

Save the complete file to a location of your choosing and record the path; it will be referenced below. In addition to not including llm_api_setup.sh in version control (e.g., .git), as a best practice, it is further recommended to not put the file in a location accessible to an LLM agent.

> [!IMPORTANT]
> If using Azure, you must choose OpenAI deployments corresponding to the specific model gpt-5-2025-08-07 for GPT5_2025_08_07_AZURE_DEPLOYMENT_NAME, otherwise the SDM estimator will have undefined behavior, since it is calibrated against that specific version (along with gemini-2.5-pro and granite-3.3-8b-instruct). We will provide new SDM estimators as new models emerge, as needed. If you have a particular enterprise need for alternative models in the near term, contact us.

> [!TIP]
> Each time you call the main Reexpress tool, gpt-5-2025-08-07 will be called 1 time and gemini-2.5-pro will be called 1 time. These calls are handled in the file [code/reexpress/mcp_utils_llm_api.py](code/reexpress/mcp_utils_llm_api.py). As with using these exact release dates of the models, we also recommend against changing the parameters of the API calls, as the behavior of the SDM estimator would then become undefined relative to its initial calibration.

## 6. Configure the MCP Server

In ${REEXPRESS_MCP_SERVER_REPO_DIR} (i.e., the repo directory) is a template file `setup/claude_desktop_config.json`:

```json
{
    "mcpServers": {
        "reexpress": {
            "type": "stdio",
            "command": "/bin/bash",
            "args": [
                "-c",
                "source /path/to/your_anaconda3_directory/etc/profile.d/conda.sh && conda activate re_mcp_v120 && source /path/to/your/llm_api_setup.sh && python /path/to/the/repo/code/reexpress/reexpress_mcp_server.py"
            ]
        }
    }
}
```

Replace the following paths with your actual paths:
- `/path/to/your/llm_api_setup.sh`: Path to your API setup script from above
- `/path/to/the/repo`: Path to the root directory of this repository (i.e., ${REEXPRESS_MCP_SERVER_REPO_DIR})
- `/path/to/your_anaconda3_directory/`: Parent path for conda, from above. As noted above, you can get this via `echo $(conda info --base)`

> [!IMPORTANT]
> The MCP server will not work if those paths are relative. They MUST be absolute paths.

For example, with  
- macOS username `a` 
- REEXPRESS_MCP_SERVER_REPO_DIR='/Users/a/Documents/repos_agents/reexpress_mcp_server'
- /Users/a/Documents/settings/llm_api_setup.sh

the file would be the following:

```json
{
    "mcpServers": {
        "reexpress": {
            "type": "stdio",
            "command": "/bin/bash",
            "args": [
                "-c",
                "source /Users/a/anaconda3/etc/profile.d/conda.sh && conda activate re_mcp_v120 && source /Users/a/Documents/settings/llm_api_setup.sh && python /Users/a/Documents/repos_agents/reexpress_mcp_server/code/reexpress/reexpress_mcp_server.py"
            ]
        }
    }
}
```

Move the updated `claude_desktop_config.json` file to `~/Library/Application\ Support/Claude/`, or merge the JSON with an existing claude_desktop_config.json file.

> [!TIP]
> You can typically use that same JSON config with other MCP clients, as well. For example, in VSCode with Github Copilot, go to Settings->Settings->search for "mcp" and supply the above JSON in settings.json. Note that only one MCP client should be open at a time with the Reexpress MCP server activated, otherwise errors may occur. 

## 7. Configure settings (optional)

See [CONFIG.md](CONFIG.md).

## Troubleshooting

If you run into issues, the first things to check are:
- Verify you are using macOS 15 on Apple silicon and have installed Claude Desktop.
- You used absolute paths in all of the above.
- You supplied valid API keys in llm_api_setup.sh, and if using Azure, your deployments are properly configured.
- Your Mac needs to be able to run "ibm-granite/granite-3.3-8b-instruct" locally. Try a simple example from the model page at [https://huggingface.co/ibm-granite/granite-3.3-8b-instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct).
- Check that there isn't something wrong with your Claude Desktop install. Try walking through the weather MCP server demo at [https://modelcontextprotocol.io/quickstart/server](https://modelcontextprotocol.io/quickstart/server). If that does not work, then other MCP servers are unlikely to work.
- Verify you are using the dependencies noted above. The underlying SDM estimator code is dependent on these versions of Faiss, NumPy, and PyTorch.
