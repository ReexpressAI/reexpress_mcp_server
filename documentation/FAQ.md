# Reexpress MCP Server FAQ

- Can I use another LLM as the tool-calling LLM?
  - Our current recommendation is to re-cross-encode the output from other LLMs with Claude, using it to then call the Reexpress tool.

- How much will the LLM API calls cost?
  - It depends on your workflow, but they are otherwise standard calls to GPT-4.1 (1 call), o4-mini (1 call), and text-embedding-3-large (2 calls over relatively short input texts) that you can budget as with other calls. The total output tokens for each LLM call tends to be relatively modest (see [/code/reexpress/mcp_utils_llm_api.py](/code/reexpress/mcp_utils_llm_api.py)); the input tokens and total number of calls will depend on your setting.

- Mac requirements
  - The lowest-spec'd Mac we have tested on to date is a M1 Max with 64 GB of unified memory, for which the on-device calculations are at interactive speed and the overall tool-call time is dominated by the LLM API calls. In principle, less memory than that is required, but has not yet been tested.

- Can I just flip in a different LLM in `code/reexpress/mcp_utils_llm_api.py`?
  - No, not with the trained SDM estimator we have provided. The provided SDM estimator is predicated on those models (and specifically, those release versions/dates and those parameter settings for the API calls). Using a different LLM, even the same model but of a different release date, would cause the behavior of the estimator to be undefined. However, that is not a fundamental limitation: Other underlying LLMs can be used, but then you need to re-train the SDM estimator. The code to do so is already in this repo, but we will provide additional instructions on how to do so in a future update.

- Can I introspect the training and calibration set, relative to the prediction, to see the text of the nearest matches?
  - That capability is not currently available, but we are working on separate software/tooling to enable that, as was possible with the on-device `Reexpress one` macOS application.

> [!TIP]
> Are you interested in adapting this approach at scale to your enterprise, domain-specific agent task? We can help you retrain the underlying SDM estimator against your data to increase the proportion of high-probability verifications. Contact us.
