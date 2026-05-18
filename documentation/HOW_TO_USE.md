# Reexpress MCP Server User Guide: User Instructions, Prompts, and Functions

## Table of Contents

1. [Getting Started](#getting-started-choose-an-mcp-client-setup-azureopenai-and-google-gemini-api-keys)
2. Tools
   - Main
     - [Reexpress](#the-reexpress-tool-reexpressuser_question-str-ai_response-str)
   - Helper tools
     - [ReexpressView](#the-reexpressview-tool-reexpress_view)
     - [ReexpressReset](#the-reexpressreset-tool-reexpress_reset)
   - Model update tools
     - [ReexpressAddFalse](#the-reexpressaddfalse-tool-reexpress_add_false)
     - [ReexpressAddTrue](#the-reexpressaddtrue-tool-reexpress_add_true)
     - [ReexpressAddOOD](#the-reexpressaddood-tool-reexpress_add_ood)
   - File Access tools (disabled by default, see [CONFIG.md](/CONFIG.md))
     - [ReexpressDirectorySet](#the-reexpressdirectoryset-tool-reexpress_directory_setdirectory-str)
     - [ReexpressFileSet](#the-reexpressfileset-tool-reexpress_file_setfilename-str)
     - [ReexpressFileClear](#the-reexpressfileclear-tool-reexpress_file_clear)

## Getting Started: Choose an MCP client; Setup Azure/OpenAI and Google Gemini API keys

The Reexpress MCP server works with any [MCP client](https://modelcontextprotocol.io/clients). Our recommended way to get started is with the [Claude Desktop App](https://claude.ai/download) for macOS Tahoe 26, running on an Apple silicon Mac, since it has web-search (which we highly recommend for verification) built-in as an option. We will assume you have downloaded and installed Claude Desktop in the following. Consult the Anthropic documentation for details.

Separately you will also need an OpenAI or Azure OpenAI API key, and a Google Gemini API key, as detailed in [INSTALL.md](/INSTALL.md).

## Using the tools

It's easy to run verification via a simple prompt. `reexpress(user_question: str, ai_response: str)` is the primary tool, and can be accessed via the recommended prompt below. We also provide tools for dynamically modifying the estimator's training set, and controlling what files the Reexpress tool sees. All of the available tools are defined in [code/reexpress/reexpress_mcp_server.py](/code/reexpress/reexpress_mcp_server.py) with the `@mcp.tool()` decorator.

## The Reexpress tool: `reexpress(user_question: str, ai_response: str)`

At the end of your prompt (or alone if referencing a previous assistant response) add:

> ```Please verify your final answer with the Reexpress tool. Do not include the Reexpress instructions themselves as part of the user_question argument of the Reexpress tool. Structure your input to the ai_response argument as Reference [Number, source or "internal knowledge" or "internal reasoning"; and if applicable, URL or filename, and the doc_index-sentence_index]: Source text; Reference [Number+1, source or "internal knowledge" or "internal reasoning"; and if applicable, URL or filename, and the doc_index-sentence_index]: Source text; Answer: Your answer. Consider your final answer verified if <successfully_verified> True </successfully_verified> and <confidence> >= 90% </confidence>.```

> [!TIP]
> On macOS you can add keyboard shortcuts (e.g., for use in Claude Desktop) in System Settings > Keyboard > Text Replacements. You can then type the short text and press SPACE for the replacement. We use the shortcut `r:r` for the main prompt above. We include what we use for each of the tools below, in turn.[^3]

This will take your previous question and Claude's response, and then ensemble it against 1 call to gpt-5.5-2026-04-23; 1 call to gemini-3.1-pro-preview; and 1 call to gemini-embedding-2, the output over which we then run the on-device SDM estimator to calculate a verification classification, where True indicates that the estimator can verify that the response answered the query or instruction, and False indicates that the estimator cannot verify that the response answered the query or instruction, at least given the provided context.

### *The key information the tool will tell you:*
1. Is the response successfully verified (a binary classification): True or False
2. Confidence in the prediction (i.e., the probability estimated for the above binary classification):
	- `>= 90%`
   - ` <= 89% (use with caution) `
   - ` < 60% (approaching random chance, so use with caution) `
   - ` Out-of-distribution (unreliable) `
3. Informal explanations from each of the API LMs. (1) and (2) above constitute the final verification decision, whereas these explanations are *inputs* to that classifier. Downstream LMs can use these explanations as additional signals, but to avoid conflating these inputs with the final classification decision, we recommend telling the tool-calling LM to do the following: `Consider your final answer verified if <successfully_verified> True </successfully_verified> and <confidence> >= 90% </confidence>.` 

### Example

Our example below is with the simple question: "What is the derivative of ln(x)?"[^1]

![Example of output from the Reexpress tool for a simple question using Claude Desktop as the MCP client.](/documentation/example_output/reexpress_tool.png)

Additionally, the HTML page at [example_output/current_reexpression.html](example_output/current_reexpression.html) is the corresponding saved output from the tool call (if enabled, see [OUTPUT_HTML.md](OUTPUT_HTML.md)). A screenshot of the first part of that page is below.

![Screenshot image of the rendered HTML output for the example from the Reexpress tool.](/documentation/example_output/current_reexpression_as_image_top_of_page.png)


Here, we see that the SDM estimator has successfully verified Claude's answer, with a probability of at least 90% relative to our [training and calibration sets](/documentation/DATA.md). Typically, we recommend using the tool at that granularity (i.e., is the output verified at a probability of at least 90%, and if not, take additional branching action until it is). 

If we need to further understand the calculation, we can call ReexpressView or look at current_reexpression.html in the model directory. The system demonstration paper and Model Card provide further details on these estimates of the predictive uncertainty.

If you scroll down to the end of current_reexpression.html, you will also see the text of the nearest match from the training/support set. This is the instance that determines the `Distance` to training, as well the `Similarity`. (In the future, we will provide additional tooling to inspect additional matches, as well as to modify the labels and delete instances, as was possible with the now deprecated `Reexpress one` macOS desktop application.)

> [!TIP]
> Out-of-the-box, the probabilities from Reexpress may initially seem conservative for your particular task. The estimates are well-calibrated against our training and calibration data, but will be cautious if your data is different from these examples. That is a unique property and benefit of an SDM estimator; alternative approaches will give over-confident probabilities when presented with distribution-shifted data, rendering the probabilities unreliable, if not altogether meaningless. Unique to the approach is that you can make local, on-the-fly adjustments based on your data with the ReexpressAddFalse, ReexpressAddTrue, and ReexpressAddOOD tools, described below.

## The ReexpressView tool: `reexpress_view()`

This will pull up details on the most recent verification, if any. Recommended prompt:

> Please use the ReexpressView tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:v`)

This provides the same streamlined output returned when calling the Reexpress tool, as well as the lower-level technical details about the estimated probability noted above (and motivated and described further in the paper linked in the README). 

> [!TIP]
> Starting in v1.1.0, the output HTML (if enabled) that is saved to current_reexpression.html in the model directory contains the information from ReexpressView, plus the nearest match from training. We generally find the HTML easier to read than scrolling through the tool output interface of typical MCP clients, and also does not require an additional tool call. For convenience, we recommend setting a bookmark in your browser to the local current_reexpression.html file. Starting in v1.2.0, the HTML also includes the brief question and response summary from GPT-5.x. This is seen by the agreement model, but we exclude it from the regular tool output to avoid confusing the tool-calling LM.

## The ReexpressReset tool: `reexpress_reset()`

Out-of-the box, the config (see [CONFIG.md](/CONFIG.md)) restricts the total number of sequential calls to the Reexpress tool to 100, before you (or the tool-calling LM, if you allow) need to call the ReexpressReset tool.

Recommended prompt:

> Please use the ReexpressReset tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:reset`)

## The ReexpressDirectorySet() tool: `reexpress_directory_set(directory: str)`

Reexpress has a simple and conservative, but effective file access convention. You can add plain text files to a
    running list. When the verification tool is called, the file content will be sent to the verification LMs. Use
    this when you want to ensure that the verification LMs have access to the verbatim text of the underlying files,
    rather than depending on the tool-calling LM (e.g., Claude) to send applicable text to the Reexpress tool. The canonical
    use-case is if you ask the tool-calling LM to analyze a document or codebase and you want the verification tool to check
    that analysis against the original document or codebase. Without this, the verification would be un-grounded
    relative to the original. On the other hand, for short code-snippets and related cases when you
    just need an initial, first-pass verification, it may be sufficient to just have the tool-calling LM send
    the relevant portions of the text to the verification tool as part of the arguments to the function
    reexpress(user_question: str, ai_response: str), as per our recommended prompt. However, in those cases,
    keep in mind that the verification is contingent on the tool-calling LM faithfully representing the grounding
    documents.
    
Use ReexpressDirectorySet() as a convenience function to optionally set a parent directory. Subsequent calls to ReexpressFileSet() can then have names or paths relative to this directory. This tool must be manually enabled in the config. See [CONFIG.md](/CONFIG.md).

Recommended prompt:

> Please use the ReexpressDirectorySet() tool, and then return directly to me without further analysis.

Include your directory (as an absolute path) within the parentheses. E.g., `ReexpressDirectorySet(/Users/a/Documents/directory_we_allow_LMs_to_see)`.

(We set our macOS keyboard Text Replacement shortcut to: `r:d`)

Tech note: File management is handled by the ExternalFileController class in [code/reexpress/mcp_utils_file_access_manager.py](/code/reexpress/mcp_utils_file_access_manager.py).

## The ReexpressFileSet tool: `reexpress_file_set(filename: str)`

See the description above for the ReexpressDirectorySet() tool. Currently only plain text files with UTF-8 encoding (e.g., .txt, .py, .swift, .md, .csv, .json, .html, etc.) are supported. This tool must be manually enabled in the config. [CONFIG.md](/CONFIG.md) describes how to control how long files will be accessible (in terms of elapsed time in minutes); the max number of lines to consider for each file; the max file size that can be opened; and the total number of files that will be considered in the running list.

Recommended prompt:

> Please use the ReexpressFileSet() tool, and then return directly to me without further analysis.

Include your filename, or absolute path with filename, within the parentheses. E.g., `ReexpressFileSet(/Users/a/Documents/directory_we_allow_LMs_to_see/file1.py)` or `ReexpressFileSet(file1.py)`, if ReexpressDirectorySet(/Users/a/Documents/directory_we_allow_LMs_to_see) has been called.

(We set our macOS keyboard Text Replacement shortcut to: `r:f`)

## The ReexpressFileClear tool: `reexpress_file_clear()`

The ReexpressFileClear tool removes all granted access to files for the verification tool. By default, if file access is enabled (see [CONFIG.md](/CONFIG.md)), all granted access to files expires after 15 minutes, but it is recommended to manually remove irrelevant files to avoid conditioning the verification on inapplicable data (and unnecessarily using LM tokens).

Recommended prompt:

> Please use the ReexpressFileClear tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:c`)

## The ReexpressAddFalse tool: `reexpress_add_false()`

If you ran the Reexpress tool, and you yourself have determined that the tool-calling LM (e.g., Claude) did NOT adequately answer your question or instruction, use this tool to update the SDM estimator by adding the labeled example to the training (support) set. (You must have write access to the model directory, otherwise this operation will fail.)

Think of this as adding a labeled example for binary classification; here, by convention 'NOT Verified' corresponds to the class at index 0 of the underlying SDM estimator. This is in terms of the original arguments to `reexpress(user_question: str, ai_response: str)`, and not to any subsequent refinement or new output of the tool-calling LM after seeing the output from the most recent Reexpress tool call. MCP clients typically have a means of checking the arguments to tools calls, which is a good idea to do each time before using this tool.  

Recommended prompt:

> Please use the ReexpressAddFalse tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:0`)

> [!TIP]
> ReexpressAddFalse, ReexpressAddTrue, and ReexpressAddOOD modify the training (support) set without re-training the estimator, which is great for fast, local updates.[^2] When you need to make a large number of updates (> 1000 examples as a rule of thumb, given the size of the base support set here), we recommend re-training/re-calibrating the estimator, which you can do using the code in this repo. All of your added data (with embedding input to the estimator) is saved to `adaptation/running_updates.jsonl` in the model directory, which you can use to re-train the estimator. Note that calls to the main Reexpress tool do not save your data to disk (beyond the one-off HTML file, if enabled, which gets overwritten each call); that only occurs if ReexpressAddFalse, ReexpressAddTrue, or ReexpressAddOOD are called.

## The ReexpressAddTrue tool: `reexpress_add_true()`

This is analogous to ReexpressAddFalse, above, but used when you yourself have determined that the tool-calling LM (e.g., Claude) DID correctly answer your question or instruction, and you want to add the example to the model.

Recommended prompt:

> Please use the ReexpressAddTrue tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:1`)

## The ReexpressAddOOD tool: `reexpress_add_ood()`

This is along the lines of ReexpressAddFalse and ReexpressAddTrue, above, but used when you yourself have determined that the output of the SDM estimator is out-of-distribution (OOD), and you want subsequent, closely related examples to match to the OOD data (and by extension, have a low Similarity value, since the model can never predict the -99 label we use for OOD data). This is a more specialized, rarely needed option. Typically, you will use the ReexpressAddFalse tool (and class label 0) as a 'sink'/catch-all class for unverified outputs for coding/data science tasks, but we provide this as an option for specialized applications. 

Recommended prompt:

> Please use the ReexpressAddOOD tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:ood`)

[^1]: In practice, with a question like this we would typically ultimately seek to call a computer algebra system. However, the limitation of existing LMs has been that we have lacked a reliable mechanism to route to such tools. With the Reexpress tool, we can route to such branching decisions via uncertainty-aware verification using an SDM estimator. In this way, although this simple example may initially seem trivial, it reflects a powerful, newfound ability to reliably classify over high-dimensional inputs that we can use to construct multi-stage LM-agent-based pipelines.

[^2]: Updating in this way changes the Similarity and Distance quantile, while the Magnitude stays fixed. This is an effective balance between fast moving and slow moving components for local updates using an SDM estimator.

[^3]: This built-in macOS text replacement feature does not work in the text input boxes of some MCP clients, nor in the Terminal.
