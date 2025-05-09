
# Reexpress Model-Context-Protocol (MCP) Server
### For Claude 3.7 Sonnet and Apple silicon on macOS Sequoia 15 (more platforms and models coming) 

### Video overview: [Here](https://youtu.be/PaWrTFPJv2M)

[![Watch the YouTube video](documentation/reexpress_mcp_server_intro_slide.png)](https://youtu.be/PaWrTFPJv2M)


Reexpress MCP Server is a drop-in solution to add state-of-the-art statistical verification to your complex LLM pipelines, as well as your everyday use of LLMs for search and QA for **software development and data science settings**. It's the first reliable, statistically robust AI second opinion for your AI workflows.

Simply install the MCP server and then add the Reexpress prompt to the end of your chat text. Claude 3.7 Sonnet (from Anthropic) will then check its response with the provided pre-trained Reexpress [Similarity-Distance-Magnitude (SDM) estimator](#citation), which ensembles gpt-4.1-2025-04-14, o4-mini-2025-04-16, and text-embedding-3-large, along with the output from Cladue 3.7, and calculates a robust estimate of the predictive uncertainty against a database of over 100,000 training examples, calibrated against over 100,000 calibration examples. Unique to the Reexpress method, you can easily adapt the model to your tasks: Simply call the ReexpressAddTrue or ReexpressAddFalse tools after a verification has completed, and then future calls to the Reexpress tool will dynamically take your updates into consideration when calculating the verification probability.

> [!NOTE]
> In addition to providing you (the user) with a principled estimate of confidence in the output given your instructions, Claude itself can use the verification output to progressively refine its answer, determine if it needs additional outside resources or tools, or has reached an impasse and needs to ask you for further clarification or information. That's what we call **reasoning with SDM verification** --- an entirely new capability in the AI toolkit that we think will open up a much broader range of use-cases for LLMs and LLM agents, for both individuals and enterprises.

Data is only sent via standard LLM API calls to Anthropic and to Azure/OpenAI; all of the processing for the SDM estimator is done locally on your Apple silicon Mac. (Optionally, we highly recommend providing access to web search via your MCP client, such as via Claude Desktop or a web-search MCP server, or for closed-domain settings, access to domain-specific retrieval.) Reexpress MCP has a simple and conservative, but effective, file access system: You control which additional files (if any) get sent to the LLM APIs by explicitly specifying files via the file-access tools ReexpressDirectorySet() and ReexpressFileSet().

## Installation

See [INSTALL.md](INSTALL.md).

> [!TIP]
> The Reexpress MCP server is straightforward to setup relative to other MCP servers, but we assume some familiarity with LLMs, MCP, and command-line tools. Our initial target audience is developers and data scientists. Only add other MCP servers from sources that you trust, and keep in mind that other MCP tools could alter the behavior of our MCP server in unexpected ways. 

## Configuration options

See [CONFIG.md](CONFIG.md).

## How to Use

See [documentation/HOW_TO_USE.md](documentation/HOW_TO_USE.md).

## Guidelines

See [documentation/GUIDELINES.md](documentation/GUIDELINES.md).

## FAQ

See [documentation/FAQ.md](documentation/FAQ.md).

## Training and Calibration Data

See [documentation/DATA.md](documentation/DATA.md).

## Citation

If you find this software useful, consider citing the following paper:

```
@misc{Schmaltz-2025-SimilarityDistanceMagnitudeUniversalVerification,
      title={Similarity-Distance-Magnitude Universal Verification}, 
      author={Allen Schmaltz},
      year={2025},
      eprint={2502.20167},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.20167}, 
}
```
