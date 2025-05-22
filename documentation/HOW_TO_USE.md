# Reexpress MCP Server User Guide: User Instructions, Prompts, and Functions

## Table of Contents

1. [Choose an MCP client](#choose-an-mcp-client-and-setup-azureopenai-api-keys)
2. Tools
   - Main
     - [Reexpress](#the-reexpress-tool-reexpressuser_question-str-ai_response-str)
   - Helper tools
     - [ReexpressView](#the-reexpressview-tool-reexpress_view)
     - [ReexpressReset](#the-reexpressreset-tool-reexpress_reset)
   - Model update tools
     - [ReexpressAddFalse](#the-reexpressaddfalse-tool-reexpress_add_false)
     - [ReexpressAddTrue](#the-reexpressaddtrue-tool-reexpress_add_true)
   - File Access tools (disabled by default, see [CONFIG.md](/CONFIG.md))
     - [ReexpressDirectorySet](#the-reexpressdirectoryset-tool-reexpress_directory_setdirectory-str)
     - [ReexpressFileSet](#the-reexpressfileset-tool-reexpress_file_setfilename-str)
     - [ReexpressFileClear](#the-reexpressfileclear-tool-reexpress_file_clear)

## Choose an MCP client and Setup Azure/OpenAI API keys

The Reexpress MCP server works with any [MCP client](https://modelcontextprotocol.io/clients). Our recommended way to get started is with the [Claude Desktop App](https://claude.ai/download) for macOS Sequoia 15, running on an Apple silicon Mac, since it has web-search (which we highly recommend for verification) built-in as an option and makes it easy to toggle extended thinking for Claude, which we recommend using when calling the main Reexpress tool. We will assume you have downloaded and installed Claude Desktop in the following. As of writing, you can get everything setup with the free plan, but you may need a paid subscription to access web search and extended thinking, which are useful for verification. Consult the Anthropic documentation for details.

Separately you will also need an OpenAI or Azure OpenAI API key, as detailed in [INSTALL.md](/INSTALL.md).

## Using the tools

It's easy to run verification via a simple prompt. `reexpress(user_question: str, ai_response: str)` is the primary tool, and can be accessed via the recommended prompt below. We also provide tools for dynamically modifying the estimators training set and controlling what files the Reexpress tool sees. All of the available tools are defined in [code/reexpress/reexpress_mcp_server.py](/code/reexpress/reexpress_mcp_server.py) with the `@mcp.tool()` decorator.

## The Reexpress tool: `reexpress(user_question: str, ai_response: str)`

At the end of your prompt (or alone if referencing a previous assistant response) add:

> Please verify your final answer with the Reexpress tool. Do not include the Reexpress instructions themselves as part of the user_question argument of the Reexpress tool. Structure your input to the ai_response argument as Reference [Number, source or "internal knowledge" or "internal reasoning"; and if applicable, URL or filename, and the doc_index-sentence_index]: Source text; Reference [Number+1, source or "internal knowledge" or "internal reasoning"; and if applicable, URL or filename, and the doc_index-sentence_index]: Source text; Answer: Your answer.

> [!TIP]
> On macOS you can add keyboard shortcuts (e.g., for use in Claude Desktop) in System Settings > Keyboard > Text Replacements. You can then type the short text and press SPACE for the replacement. We use the shortcut `r:r` for the main prompt above. We include what we use for each of the tools below, in turn. (Unfortunately, as of writing, text replacement does not work in the text input boxes of some MCP clients, such as VSCode Copilot.)

This will take your previous question and Claude's response, and then ensemble it against 1 call to gpt-4.1-2025-04-14; 1 call to o4-mini-2025-04-16 (reasoning_effort="medium"); and 2 calls to text-embedding-3-large, the output over which we then run the on-device SDM estimator to calculate a verification classification, where True indicates that the estimator can verify that the response answered the query or instruction, and False indicates that the estimator cannot verify that the response answered the query or instruction, at least given the provided context.

Our example below is with the simple question: "What is the derivative of ln(x)?"[^1]

![Example of output from the Reexpress tool for a simple question using Claude Desktop as the MCP client.](/documentation/reexpress_tool.png)

Here, we see that the SDM estimator has successfully verified Claude's answer, with a probability of 0.99 (which is the max display probability, see Footnote 1) relative to our [training and calibration set](/documentation/DATA.md). Unique to the Reexpress approach, we also provide an estimate of the reliability of the calibration process itself. Here, we summarize this with one of three values: "Highest" (the desired value)[^2]; "Low", indicating the statistical estimation process and the resulting probability might not be reliable; and "Lowest", indicating the probability is not a reliable estimate and the verification classification is out-of-distribution. 

The verification classification and its accompanying probability and estimate of calibration reliability are the source of truth with respect to the verification decision. For reference, we also provide the generated explanations from gpt-4.1-2025-04-14 ("Informal Explanation [1]") and o4-mini-2025-04-16 ("Informal Explanation [2]", which is occluded by the scroll bar in the image above), which Claude can use to help refine its answer, conditional on the classification decision.

> [!TIP]
> Out-of-the-box, the probabilities from Reexpress may initially seem relatively conservative for your particular task. The estimates are well-calibrated against our observed >200,000 training and calibration instances, but will be cautious if your data is different from these examples. That is a unique property and benefit of an SDM estimator; alternative approaches will give over-confident probabilities when presented with distribution-shifted data, rendering the probabilities meaningless. Also unique to the approach is that you can make local, on-the-fly adjustments based on your data with the ReexpressAddFalse and ReexpressAddTrue tools, described below.

## The ReexpressView tool: `reexpress_view()`

This will pull up details on the most recent verification, if any. Recommended prompt:

> Please use the ReexpressView tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:v`)

This provides the same streamlined output returned when calling the Reexpress tool, as well as lower-level technical details about the estimated probability, as described in the paper linked in the README.

## The ReexpressReset tool: `reexpress_reset()`

Out-of-the box, the config (see [CONFIG.md](/CONFIG.md)) restricts the total number of sequential calls to the Reexpress tool to 10, before you (or Claude, if you allow) need to call the ReexpressReset tool.

Recommended prompt:

> Please use the ReexpressReset tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:reset`)

## The ReexpressDirectorySet() tool: `reexpress_directory_set(directory: str)`

Reexpress has a simple and conservative, but effective file access convention. You can add plain text files to a
    running list. When the verification tool is called, the file content will be sent to the verification LLMs. Use
    this when you want to ensure that the verification LLMs have access to the verbatim text of the underlying files,
    rather than depending on the tool-calling LLM (e.g., Claude) to send applicable text to the Reexpress tool. The canonical
    use-case is if you ask an LLM to summarize a document or codebase and you want the verification tool to check
    that summary against the original document or codebase. Without this, the verification would be un-grounded
    relative to the original. On the other hand, for short code-snippets and related cases when you
    just need an initial, first-pass verification, it may be sufficient to just have the tool-calling LLM send
    the relevant portions of the text to the verification tool as part of the arguments to the function
    reexpress(user_question: str, ai_response: str), as per our recommended prompt. However, in those cases,
    keep in mind that the verification is contingent on the tool-calling LLM faithfully representing the grounding
    documents.
    
Use ReexpressDirectorySet() as a convenience function to optionally set a parent directory. Subsequent calls to ReexpressFileSet() can then have relatives names or paths relative to this directory. This tool must be manually enabled in the config. See [CONFIG.md](/CONFIG.md).

Recommended prompt:

> Please use the ReexpressDirectorySet() tool, and then return directly to me without further analysis.

Include your directory (as an absolute path) within the parentheses. E.g., `ReexpressDirectorySet(/Users/a/Documents/directory_we_allow_LLMs_to_see)`.

(We set our macOS keyboard Text Replacement shortcut to: `r:d`)

Tech note: File management is handled by the ExternalFileController class in [code/reexpress/mcp_utils_file_access_manager.py](/code/reexpress/mcp_utils_file_access_manager.py).

## The ReexpressFileSet tool: `reexpress_file_set(filename: str)`

See the description above for the ReexpressDirectorySet() tool. Currently only plain text files with UTF-8 encoding (e.g., .txt, .py, .swift, .md, .csv, .json, .html, etc.) are supported. This tool must be manually enabled in the config. [CONFIG.md](/CONFIG.md) describes how to control how long files will be accessible (in terms of elapsed time in minutes); the max number of lines to consider for each file; the max file size that can be opened; and the total number of files that will be considered in the running list.

Recommended prompt:

> Please use the ReexpressFileSet() tool, and then return directly to me without further analysis.

Include your filename, or absolute path with filename, within the parentheses. E.g., `ReexpressFileSet(/Users/a/Documents/directory_we_allow_LLMs_to_see/file1.py)` or `ReexpressFileSet(file1.py)`, if ReexpressDirectorySet(/Users/a/Documents/directory_we_allow_LLMs_to_see) has been called.

(We set our macOS keyboard Text Replacement shortcut to: `r:f`)

## The ReexpressFileClear tool: `reexpress_file_clear()`

The ReexpressFileClear tool removes all granted access to files for the verification tool. By default, if file access is enabled (see [CONFIG.md](/CONFIG.md)), all granted access to files expires after 5 minutes, but it is recommended to manually remove irrelevant files to avoid conditioning the verification on inapplicable data (and unnecessarily using LLM tokens).

Recommended prompt:

> Please use the ReexpressFileClear tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:c`)

## The ReexpressAddFalse tool: `reexpress_add_false()`

If you ran the Reexpress tool, and you yourself have determined that the tool-calling LLM (e.g., Claude) did NOT adequately answer your question or instruction, use this tool to update the SDM estimator by adding the labeled example to the training (support) set. (You must have write access to the model directory, otherwise this operation will fail.)

Think of this as adding a labeled example for binary classification; here, by convention 'NOT Verified' corresponds to the class at index 0 of the underlying SDM estimator. This is in terms of the original arguments to `reexpress(user_question: str, ai_response: str)`, and not to any subsequent refinement or new output of the tool-calling LLM after seeing the output from the most recent Reexpress tool call. MCP clients typically have a means of checking the arguments to tools calls, which is a good idea to do each time before using this tool.  

Recommended prompt:

> Please use the ReexpressAddFalse tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:0`)

> [!TIP]
> ReexpressAddFalse and ReexpressAddTrue modify the training (support) set without re-training the estimator, which is great for fast, local updates.[^3] After around 10,000 updates (or if you have a very unique, domain-specific setting), it can be a good idea to altogether re-train the estimator. The code to do so is already in this repo, but we will provide detailed instructions on how to do so in a future update. All of your added data (with embedding input to the estimator) is saved to `adaptation/running_updates.jsonl` in the model directory, which you can use to re-train the estimator. Note that calls to the main Reexpress tool do not save your data to disk; that only occurs if ReexpressAddFalse or ReexpressAddTrue are called.

## The ReexpressAddTrue tool: `reexpress_add_true()`

This is analogous to ReexpressAddFalse, above, but used when you yourself have determined that the tool-calling LLM (e.g., Claude) DID correctly answer your question or instruction, and you want to add the example to the model.

Recommended prompt:

> Please use the ReexpressAddTrue tool, and then return directly to me without further analysis.

(We set our macOS keyboard Text Replacement shortcut to: `r:1`)

[^1]: In practice, with a question like this we would typically ultimately seek to call a computer algebra system, with each of those branching decisions determined via verification using the Reexpress tool. For reference, the result on this run when calling ReexpressView on this statement is that the conservative p(y|x)\_lower probability is 0.999000731695713 (i.e., very close to 1), with an effective sample size of 35,715 (i.e., very large), in the 0.995 distance quantile (i.e., very close to the observed training manifold), and the estimate is valid index-conditional calibrated (i.e., prediction- and class-conditional calibrated, here at an alpha' of 0.95). That is, although this simple example may initially seem trivial, in fact it reflects a powerful, newfound ability to reliably classify over high-dimensional inputs that we can use to construct multi-stage LLM-agent-based pipelines.

[^2]: "Highest" calibration reliability corresponds to a prediction that is index-conditional calibrated at an alpha' value of 0.95. See the cited paper for details. Since existing tool-calling LLMs do not have a notion of these epistemic uncertainty quantities, we simplify the output by setting a ceiling of 94% on non-index-conditional estimates, and we pin classification and distribution mismatches to parity (i.e., 50%). The calibrated probability is always in terms of the verified class (class 1), but the estimate of calibration reliability is distribution-wide; that is, the calibration reliability can be "Highest" with a low (<= 0.05) probability for class 1, indicating the estimator is reliably confident the class is 0 (i.e., "NOT verified"). These transforms occur in [code/reexpress/mcp_utils_test.py](../code/reexpress/mcp_utils_test.py) -> `format_sdm_estimator_output_for_mcp_tool()`. For presentation, all probabilities in the main output are clamped between 0.01 and 0.99, with a resolution of 0.01.

[^3]: Updating in this way changes the Similarity and Distance quantile, while the Magnitude stays fixed. This is an effective balance between fast moving and slow moving components for an SDM estimator.
