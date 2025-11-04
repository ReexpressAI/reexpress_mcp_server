# Reexpress MCP Server FAQ

- Can I use another LLM as the tool-calling LLM?
  - In principle yes, but our current recommendation is to re-cross-encode the output from other LLMs with Claude, using it to then call the Reexpress tool.

- How much will the LLM API calls cost?
  - It depends on your workflow, but they are otherwise standard calls to GPT-5 (1 call) and Gemini-2.5-pro (1 call) that you can budget as with other calls. The total output tokens for each LLM call tends to be relatively modest (see [/code/reexpress/mcp_utils_llm_api.py](/code/reexpress/mcp_utils_llm_api.py)); the input tokens and total number of calls will depend on your setting.

- Mac requirements
  - The lowest-spec'd Mac we have tested on to date is a M1 Max with 64 GB of unified memory, for which the on-device calculations of the SDM estimator are at interactive speed and the overall tool-call time is dominated by the LLM API calls and the generation of 1 token from the 8-billion-parameter local LLM. In principle, less memory than that is required, but has not yet been tested by us internally.

- Can I just flip in a different LLM in `code/reexpress/mcp_utils_llm_api.py`?
  - No, not with the trained SDM estimator we have provided. The provided SDM estimator is predicated on those models (and specifically, those release versions/dates and those parameter settings for the API calls). Using a different LLM, even the same model but of a different release date, would cause the behavior of the estimator to be undefined. However, that is not a fundamental limitation: Other underlying LLMs can be used, but then you need to re-train the SDM estimator. The code to do so is in this repo. The training script in the model directory, which is also included at `documentation/model_details/release/v2.0.0/train_and_eval_sdm_estimator_v2.0.0.sh`, can be used as a guide.

- Can I introspect the training/support set, relative to the prediction, to see the text of the nearest match?
  - Yes! Viewing the first match in the support set is enabled starting in version 1.1.0 if you enable generating the static HTML for each tool call. See documentation/OUTPUT_HTML.md. (In the future, we will provide additional tooling to inspect additional matches, as well as to modify the labels and delete instances, and for hard-attention-based feature detection, as was possible with the now deprecated `Reexpress one` macOS desktop application.)

> [!TIP]
> Are you interested in adapting this approach at scale to your enterprise, domain-specific agent task? We can help you retrain the underlying SDM estimator against your data to increase the proportion of high-probability verifications. Contact us.
