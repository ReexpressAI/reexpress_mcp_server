# Changelog: Reexpress Model-Context-Protocol (MCP) Server

## What's new in version 2.0.0

Version 2.0.0 introduces our updated formulation of the SDM estimator. The SDM activation function remains the same, but the calibration method for the SDM estimator is simplified while retaining the desirable properties of the earlier version that had an additional rescaling transform. You can read about this version in our publications (see the citations included in README.md). Moving forward, the convention is to refer to this version as the canonical "SDM estimator".

Separately, we have also refactored and rewritten the code to dramatically improve efficiency, enabling scaling to much larger datasets and training SDM language models, the code for which is also included here (see our paper below and the separate research [repo](https://github.com/ReexpressAI/sdm_activations) for details). 

## What's new in version 1.2.0

Version 1.2.0 replaces the calls to gpt-4.1-2025-04-14 and o4-mini-2025-04-16-high with a single call to gpt-5-2025-08-07. Consistent with the behavior of an SDM estimator, the earlier versions using the weaker models as inputs were also well-calibrated, but the addition of GPT-5 leads to a noticeable increase in the proportion of non-rejected documents over the held-out test sets. We have additionally updated the [OpenVerification1](https://huggingface.co/datasets/ReexpressAI/OpenVerification1) dataset with the new examples.

## What's new in version 1.1.0

Version 1.1.0 adds a number of new capabilities:

- We added gemini-2.5-pro as part of the model ensemble.
- We increased the o4-mini-2025-04-16 reasoning budget from medium to high.
- We replaced the API calls to the text-embedding-3-large embeddings model with the locally run `ibm-granite/granite-3.3-8b-instruct` model, which we use to construct the representation space over the model explanations from gpt-4.1-2025-04-14, o4-mini-2025-04-16-high, and gemini-2.5-pro.
- We added the ability to *introspect* the predictions against the training set. You can now view the nearest match to each test instance via a static webpage that you can (optionally) generate for each prediction. This also makes it easy to quickly check how the verification estimation was determined without having to call the ReexpressView tool. See [documentation/OUTPUT_HTML.md](documentation/OUTPUT_HTML.md) for examples.
- We include the training script for the model and the model evaluation outputs over the OpenVerification1 dataset in the model directory (see the Release archive). A summary of the evaluation is available at [documentation/EVAL.md](documentation/EVAL.md).
- The training and calibration data are a subset of the full [OpenVerification1](https://huggingface.co/datasets/ReexpressAI/OpenVerification1) dataset, which we have made available on HuggingFace datasets.
- We have updated the output to the MCP server to have all content returned within XML tags to simplify use out-of-the-box for downstream, test-time search graphs. We have also updated our recommended base tool-call prompt with the following final sentence: `Consider your final answer verified if <successfully_verified> True </successfully_verified> and <confidence> >= 90% </confidence>.` 
- We have simplified the presentation of the verification confidence (i.e., the probability estimated for the binary classification prediction) in the main output to the following three bins to reflect the resolution at which we recommend using the tool:
    - `>= 90%`
    - `< 90% (use with caution)`
    - `Out-of-distribution (unreliable)`

- Note that we have reduced the probability threshold to 0.9 (i.e., alpha'=0.9, down from the more stringent 0.95 in version 1.0.0) to better reflect the capabilities of the current generation of models and the intended use-case of verification with a human-in-the-loop. This version admits approximately 62% of in-distribution examples at alpha'=0.9 (i.e., the proportion of valid index-conditional estimates at alpha'=0.9) from the 5k test set of the OpenVerification1 dataset, over which the marginal accuracy is approximately 92%. If you need a version with a more stringent requirement (and/or recalibration over your domain specific tasks), we provide the training code here, as noted above. For mission-critical enterprise settings and semi-autonomous agents that require `alpha' > 0.9`, we typically recommend training a full SDM network that composes the hidden states over all input text (prompt, response, and if applicable, the composition of the output of additional LLMs). (In contrast, the current MCP server uses an SDM estimator that marginalizes over the content of the prompt and response, and takes as input an ensemble of explanations from external LLMs. This is done to keep computational costs manageable for local deployment with existing LLM APIs.) We can assist you with building such SDM networks. Contact us!
- We modified the baseline configuration in [code/reexpress/mcp_settings.json](code/reexpress/mcp_settings.json).
- Finally, we added a new tool function, reexpress_add_ood(), which allows you to add an out-of-distribution (label=-99) instance to the support set. (For developers and researchers, we have also updated the training and calibration routines to allow such instances to participate in training and calibration. Instances in the --ood_support_file get added to the training support for each training iteration, and thus can impact the Similarity values of training and calibration instances, if applicable.)
- For researchers: Starting in commit c816516 is a script (`utils_graph_output.py`) to construct interactive graphs of the batch output. See the end of the [training script](documentation/model_details/release/v1.1.0/train_and_eval_sdm_estimator_v1.1.0.sh) for example use. You can click on a point to print additional information to the console. [Graphs for the output](documentation/model_details/release/v1.1.0/output_graphs) of the SDM estimator in this release are saved to the repo for reference.
