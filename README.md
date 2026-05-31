# Reexpress Model-Context-Protocol (MCP) Server
### For tool-calling LLMs (e.g., Claude Opus 4.7) and MCP clients running on macOS (Tahoe 26 or later on Apple silicon) or Linux

### Video overview[^1]: [Here](https://youtu.be/PaWrTFPJv2M)

[![Watch the YouTube video](documentation/reexpress_mcp_server_intro_slide.png)](https://youtu.be/PaWrTFPJv2M)

![Screenshot image of the rendered HTML output from the Reexpress tool.](documentation/example_output/html_output_examples/current_reexpression_pos_example_as_image.png)

![Re](documentation/fun/re.jpeg)

Reexpress MCP Server is a drop-in solution to add state-of-the-art statistical verification to your complex LLM pipelines, as well as your everyday use of LLMs for search and QA for **software development and data science settings**. It's the first reliable, statistically robust AI second opinion for your AI workflows.

Simply install the MCP server and then add the Reexpress prompt to the end of your chat text. The tool-calling LLM (e.g., Anthropic's LLM model Claude Opus 4.7) will then check its response with the provided pre-trained Reexpress [Similarity-Distance-Magnitude (SDM) estimator](#citation), which ensembles gpt-5.5-2026-04-23, gemini-3.1-pro-preview, and gemini-embedding-2, along with the output from the tool-calling LLM, and calculates a robust estimate of the predictive uncertainty against a database of training and calibration examples from the OpenVerification1 dataset. Unique to the Reexpress method, you can easily adapt the model to your tasks: Simply call the ReexpressAddTrue or ReexpressAddFalse tools after a verification has completed, and then future calls to the Reexpress tool will dynamically take your updates into consideration when calculating the verification probability. We also include the training scripts for the model, so that you can run a full retraining when more substantive changes are needed, or you want to use alternative underlying LLMs.

> [!NOTE]
> In addition to providing you (the user) with a principled estimate of confidence in the output given your instructions, the tool-calling LLM itself can use the verification output to progressively refine its answer, determine if it needs additional outside resources or tools, or has reached an impasse and needs to ask you for further clarification or information. That's what we call **reasoning with SDM verification** --- an entirely new capability in the AI toolkit that we think will open up a much broader range of use-cases for LLMs and LLM agents, for both individuals and enterprises.

Data is only sent via standard LLM API calls to Azure/OpenAI and Google, with the gemini-3.1-pro-preview calls given standard web search access through the API; all of the processing for the SDM estimator is done locally on your computer. Reexpress MCP has a simple and conservative, but effective, file access system: You control which additional files (if any) get sent to the LLM APIs by explicitly specifying files via the file-access tools ReexpressDirectorySet() and ReexpressFileSet().

## What's new in version 2.4.0

The model card is available [here](documentation/model_cards/model_card_v240.pdf).

Version 2.4.0 uses gpt-5.5-2026-04-23 and gemini-3.1-pro-preview as the generative models. As with 2.3.0.preview, gemini-embedding-2 replaces the local granite-3.3-8b-instruct model as the agreement representation model. This greatly simplifies running the Server, since you no longer need to locally run a multi-billion parameter model. Additionally, we have also expanded the OpenVerification1 dataset with new examples. See the [model card](documentation/model_cards/model_card_v240.pdf) for details. 

Additional notes in [changelog.md](changelog.md). 

## System Requirements

The MCP server runs on Linux and macOS. The primary requirement is that the machine running the MCP server needs to be able to locally run a small 3 million parameter PyTorch model, so the compute requirements are minimal. (That is as written: Only 3 *million* parameters; not 3 *billion* parameters. The model consists of an SDM activation over gemini-embedding-2 and the classification output of the two API language models.)

## Installation

See [INSTALL.md](INSTALL.md).

> [!TIP]
> The Reexpress MCP server is straightforward to setup relative to other MCP servers, but we assume some familiarity with LLMs, MCP, and command-line tools. Our target audience is developers and data scientists. Only add other MCP servers from sources that you trust, and keep in mind that other MCP tools could alter the behavior of our MCP server in unexpected ways. 

## Configuration options

See [CONFIG.md](CONFIG.md).

## How to Use

See [documentation/HOW_TO_USE.md](documentation/HOW_TO_USE.md).

## Generating static HTML with output from the tool call

See [documentation/OUTPUT_HTML.md](documentation/OUTPUT_HTML.md).

## Guidelines

See [documentation/GUIDELINES.md](documentation/GUIDELINES.md).

## FAQ

See [documentation/FAQ.md](documentation/FAQ.md).

## Training and Calibration Data

See [documentation/DATA.md](documentation/DATA.md).

## Evaluation over OpenVerification1

See [documentation/EVAL.md](documentation/EVAL.md).

## System Demonstration Paper

A copy of our system demonstration paper "Introspectable, Updatable, and Uncertainty-aware Classification of Language Model Instruction-following", which focuses in particular on version 2.1.0 of the Reexpress MCP Server, is included [here](documentation/system_demonstration_paper/sdm_demo.pdf). The support scripts to replicate the analysis are included [here](documentation/model_details/release/v2.1.0_demo_paper).

The model card for version 2.4.0, which highlights the changes since the system demonstration paper, is available [here](documentation/model_cards/model_card_v240.pdf).

![CAIS 2026 system demonstration poster.](documentation/system_demonstration_paper/poster/reexpress_mcp_server_demo_paper_poster_cais_2026.png)

## Citation

If you find this software useful, consider citing the following peer-reviewed papers:

```
@misc{Schmaltz-2025-SimilarityDistanceMagnitudeActivations,
      title={Similarity-Distance-Magnitude Activations}, 
      author={Allen Schmaltz},
      year={2025},
      eprint={2509.12760},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.12760},
      note={To appear in \emph{Findings of the Association for Computational Linguistics: ACL 2026}, San Diego, CA, USA.},
}
```

```
@inproceedings{Schmaltz-2026-ReexpressMCPServer,
author = {Schmaltz, Allen},
title = {Introspectable, Updatable, and Uncertainty-aware Classification of Language Model Instruction-following},
year = {2026},
isbn = {9798400724152},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3786335.3813214},
doi = {10.1145/3786335.3813214},
abstract = {In this system demonstration paper, we introduce an open-source implementation for training and testing Similarity-Distance-Magnitude (SDM) estimators for the task of binary classification of instruction-following of closed-weight language models (LMs). This SDM estimator provides an approximately conditional estimate of the predictive uncertainty over instruction-following, conditional on multiple closed-weight LMs and the representation space of an open-weight model. While it would be more robust to use as input to the SDM estimator the hidden-states of the underlying models, this indirect, compositional proxy is more reliable than verbalized uncertainty and adds a means of auditing the predictions against data with known labels. We release the code as an MCP Server to simplify adding interpretability-by-exemplar and locally updatable, uncertainty-aware instruction-following to agent-based pipelines. We further release OpenVerification1, a balanced set of over two million examples of instruction-following and associated rationales from recent closed-weight LMs, for bootstrapping domain-specific estimators. Finally, we discuss limitations of estimating the predictive uncertainty without access to the hidden-states of the tool-calling LM and provide practical guidance for applications.},
booktitle = {Proceedings of the ACM Conference on AI and Agentic Systems},
pages = {1259–1269},
numpages = {11},
keywords = {Approximately conditional calibration, Interpretability-by-exemplar, Classification of instruction-following, Model ensembles},
location = {
},
series = {CAIS '26}
}
```

[^1]: The output format has changed since v1.0.0 used in the video. See [changelog.md](changelog.md).
