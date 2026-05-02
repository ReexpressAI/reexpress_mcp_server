
# Installation Instructions for the Reexpress Model-Context-Protocol (MCP) Server
### For tool-calling LLMs (e.g., Claude Opus 4.7) and MCP clients running on macOS (Tahoe 26 or later on Apple silicon) or Linux

The Reexpress MCP server works with any [MCP client](https://modelcontextprotocol.io/clients). The easiest way to get started is with the [Claude Desktop App](https://claude.ai/download) for macOS Tahoe 26, running on an Apple silicon Mac, since it has web-search (which we highly recommend for verification) built-in as an option. We will assume you have downloaded and installed Claude Desktop in the following.

Installation consists of installing conda, downloading the repo AND model file, installing Python dependencies, and setting file paths. You will also need an [OpenAI API key](https://platform.openai.com/api-keys) or [Microsoft Azure](https://azure.microsoft.com) OpenAI model deployments and corresponding API key (as described below), and a [Google Gemini API key](https://aistudio.google.com/).

## 1. Install Conda

If you don't have Conda installed, download and install Anaconda, Miniconda, or Miniforge:
- [Conda](https://docs.conda.io/en/latest/)
 
After installation, open your terminal and run:

```bash
echo $(conda info --base)
``` 

This should print something like `/Users/YOUR_USER_NAME/miniconda3` (or miniforge3, anaconda3, or your chosen install path, or /opt/conda or similar on Linux). We will refer to this as your conda path; you'll need it later.

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

### macOS Tahoe 26 and Apple silicon:

```
conda create -n re_mcp_v230 python=3.12
conda activate re_mcp_v230
conda install -c pytorch -c conda-forge \
  faiss-cpu=1.14.1 \
  pytorch=2.9.1
python -m pip install \
  matplotlib==3.10.9 \
  "mcp[cli]==1.26.0" \
  openai==2.32.0 \
  google-genai==1.73.1
```

The above will create an environment "re_mcp_v230" with the required dependencies for macOS Tahoe 26 and Apple silicon.

### Linux with Nvidia GPUs

For running the train/eval script reexpress.py on Nvidia GPUs to recalibrate over new data (i.e., create a new SDM estimator), the following environment has been verified to work on Nvidia L4 GPUs:

```
conda create -n re_mcp_v230 python=3.12
conda activate re_mcp_v230
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.14.1
python -m pip install \
  torch==2.9.1 \
  matplotlib==3.10.9 \
  "mcp[cli]==1.26.0" \
  openai==2.32.0 \
  google-genai==1.73.1
```

For some cuda drivers and setups, it may be necessary to build Faiss from source. See https://github.com/facebookresearch/faiss/blob/main/INSTALL.md for details.

Note that reexpress.py also works with --main_device="cpu" and --main_device="mps". In contrast, the MCP Server (reexpress_mcp_server.py) is currently simply hardcoded to use cpu given the typically lightweight compute requirements at test-time.

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
# Fill in with your deployment name for GPT-5.4. Replace this with whatever name you chose in Azure.
export GPT_5_4_MODEL_2026_03_05_AZURE_DEPLOYMENT='gpt-5.4'

# Google Gemini API key is required
export GEMINI_API_KEY='REPLACE_WITH_API_KEY'

# '1' to create the HTML page in ${REEXPRESS_MCP_MODEL_DIR}/visualize/current_reexpression.html for each call to the MCP server; '0' turns this feature off
export REEXPRESS_MCP_SAVE_OUTPUT='0'
```

Replace REEXPRESS_MCP_SERVER_REPO_DIR and REEXPRESS_MCP_MODEL_DIR with the repo and model directory from above, respectively. If you want to use OpenAI, set USE_AZURE_01='0' and supply your API key in OPENAI_API_KEY. If instead, you want to use Azure, set USE_AZURE_01='1' and fill in the corresponding 3 variables. Optionally add a Huggingface model cache directory.

Set REEXPRESS_MCP_SAVE_OUTPUT='1' if you want to create an HTML page for the tool call output.

Save the complete file to a location of your choosing and record the path; it will be referenced below. In addition to not including llm_api_setup.sh in version control (e.g., .git), as a best practice, it is further recommended to not put the file in a location accessible to an LLM agent.

> [!IMPORTANT]
> If using Azure, you must choose OpenAI deployments corresponding to the specific model gpt-5.4-2026-03-05 for GPT_5_4_MODEL_2026_03_05_AZURE_DEPLOYMENT, otherwise the SDM estimator will have undefined behavior, since it is calibrated against that specific version (along with gemini-3.1-pro-preview and gemini-embedding-2). We will provide new SDM estimators as new models emerge, as needed. If you have a particular enterprise need for alternative models in the near term, contact us.

> [!TIP]
> Each time you call the main Reexpress tool, gpt-5.4-2026-03-05 will be called 1 time, gemini-3.1-pro-preview, and gemini-embedding-2 will be called 1 time. These calls are handled in the file [code/reexpress/mcp_utils_llm_api.py](code/reexpress/mcp_utils_llm_api.py). As with using these exact release dates of the models, we also recommend against changing the parameters of the API calls, as the behavior of the SDM estimator would then become undefined relative to its initial calibration.

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
                "source /path/to/your_conda_directory/etc/profile.d/conda.sh && conda activate re_mcp_v230 && source /path/to/your/llm_api_setup.sh && python /path/to/the/repo/code/reexpress/reexpress_mcp_server.py"
            ]
        }
    }
}
```

Replace the following paths with your actual paths:
- `/path/to/your/llm_api_setup.sh`: Path to your API setup script from above
- `/path/to/the/repo`: Path to the root directory of this repository (i.e., ${REEXPRESS_MCP_SERVER_REPO_DIR})
- `/path/to/your_conda_directory/`: Parent path for conda, from above. As noted above, you can get this via `echo $(conda info --base)`

> [!IMPORTANT]
> The MCP server will not work if those paths are relative. They MUST be absolute paths.

For example, with  
- macOS username `a` 
- REEXPRESS_MCP_SERVER_REPO_DIR='/Users/a/Documents/repos_agents/reexpress_mcp_server'
- /Users/a/local_projects/llm_api_setup_v2.3.0.sh (Here, we place llm_api_setup_v2.3.0.sh outside of the Documents folder in a directory arbitrarily named 'local_projects' to simplify macOS file permissions when using Claude for Mac.)

the file would be the following:

```json
{
    "mcpServers": {
        "reexpress": {
            "type": "stdio",
            "command": "/bin/bash",
            "args": [
                "-c",
                "source /Users/a/miniconda3/etc/profile.d/conda.sh && conda activate re_mcp_v230 && source /Users/a/local_projects/llm_api_setup_v2.3.0.sh && python /Users/a/Documents/repos_agents/reexpress_mcp_server/code/reexpress/reexpress_mcp_server.py"
            ]
        }
    }
}
```

Move the updated `claude_desktop_config.json` file to `~/Library/Application\ Support/Claude/`, or merge the JSON with an existing claude_desktop_config.json file.

> [!TIP]
> You can typically use that same JSON config with other MCP clients, as well. For example, in VSCode with Github Copilot, go to Settings->Settings->search for "mcp" and supply the above JSON in settings.json.  

## 7. Configure settings (optional)

See [CONFIG.md](CONFIG.md).

## Troubleshooting

If you run into issues, the first things to check are:
- Verify you are using macOS Tahoe 26 on Apple silicon and have installed Claude Desktop. (In principle, the above instructions should also work on linux distributions using alternative MCP clients available for linux. Let us know if you run into issues.)
- You used absolute paths in all of the above.
- You supplied valid API keys in llm_api_setup.sh, and if using Azure, your deployments are properly configured. Try directly calling the functions in mcp_utils_llm_api.py to check that the third-party APIs are working with your supplied keys.
- Check that there isn't something wrong with your Claude Desktop install. Try walking through the weather MCP server demo at [https://modelcontextprotocol.io/quickstart/server](https://modelcontextprotocol.io/quickstart/server). If that does not work, then other MCP servers are unlikely to work.
- Verify you are using the dependencies noted above. On macOS, if you install Faiss via conda and pytorch via pip, you can run into an issue with duplicate OpenMP runtimes leading to a crash. The current solution is to install both from conda on macOS.

## Developer tip

The MCP server code is a lightweight wrapper around the SDM estimator code, making it easy to call SDM estimators for verification using existing IDEs and MCP clients. For your own client applications, you can also just call the async functions in `reexpress_mcp_server/code/reexpress/reexpress_mcp_server.py` directly. That can be useful for building more complex test-time search strategies, with additional control over the test-time branching actions.
