# OpenVerification1 Eval

## Output Predictions and Evaluations

See the model directory `sdm_estimator__gpt5_gemini_granite8b_e100iter3_v1_2_0_r1_0.9_1000/model_details/final_eval_output` for the output predictions (`all_predictions.jsonl`), logs (`version_1.2.0.log.txt`), sorted possible label errors[^1] (`possible_label_errors.jsonl`), and sorted valid index-conditional predictions[^2] (`valid_index_conditional.jsonl`) for each eval split, as well as the model's calibration set. For reference, graphs of the output are also saved in the [output_graphs](model_details/release/v1.2.0/output_graphs) directory in this repo.

We provide high-level summary statistics here. This is an evaluation of the underlying SDM estimator used in the MCP Server; Opus 4.1 (the recommended tool-calling LLM) is not involved. In other words, the arguments to `reexpress(user_question: str, ai_response: str)` are the question and response from the benchmark dataset.

### Version 1.2.0

| Data Split | Marginal Accuracy | Admitted Accuracy | Admitted Proportion | Dataset Size |
|------------|-------------------|-------------------|---------------------|-----------|
| Calibration (not held-out) | 0.93 | 0.98 | 0.61 | 74684 |
| MMLU Validation (binary verification) | 0.93 | 0.98 | 0.47 | 3036 |
| OpenVerification1 5k Test | 0.93 | 0.99 | 0.63 | 5000 |
| MMLU-Pro-4-QA-GPT4o-Letters | 0.86 | 0.93 | 0.30 | 5346 |
| MMLU-Pro-4-QA-GPT4o-Explanations | 0.86 | 0.94 | 0.44 | 5344 |

Interestingly, with this version in particular, the models are sufficiently strong that a non-trivial number of annotation errors (i.e., irreducible/aleatoric error) is evident with the MMLU-Pro-4-QA datasets. These can be examined interactively using `utils_graph_output.py`. See the end of the [training script](model_details/release/v1.2.0/train_and_eval_sdm_estimator_v1.2.0.sh) for example use.

### Version 1.1.0

| Data Split | Marginal Accuracy | Admitted Accuracy | Admitted Proportion | Dataset Size |
|------------|-------------------|-------------------|---------------------|-----------|
| Calibration (not held-out) | 0.92 | 0.98 | 0.62 | 28225 |
| MMLU Validation (binary verification) | 0.88 | 0.97 | 0.33 | 3036 |
| OpenVerification1 5k Test | 0.92 | 0.98 | 0.62 | 5000 |
| MMLU-Pro-4-QA-GPT4o-Letters | 0.74 | 0.96 | 0.09 | 5346 |
| MMLU-Pro-4-QA-GPT4o-Explanations | 0.84 | 0.95 | 0.31 | 5344 |

Legend: The "Admitted" instances are those for which the SDM estimator assigns a valid index-conditional estimate. In the context of the MCP Server, these instances would have a confidence `>= 90%` in the main output.

As indicated in the table, the estimator correctly detects the instances that have a probability `>= 0.9`, even as the overall marginal accuracy varies substantially relative to the model's Calibration set.

[^1]: These are instances that are valid-index conditional estimates, but the predicted class does not match the ground-truth label. These are sorted descending by `p(y | x)_lower`.
[^2]: These are instances for which the confidence label of the main output from the MCP Server would be `>= 90%`. These are sorted descending by `p(y | x)_lower`.
