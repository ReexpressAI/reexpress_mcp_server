# Training and Calibration Data

In addition to our internal data, the pre-trained SDM estimator makes use of publicly available open-source data: [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M) (Apache-2.0); [HL-FEVER train](https://huggingface.co/datasets/pminervini/hl-fever) (MIT); [MMLU-Pro validation](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (MIT); and [MMLU auxiliary-train, dev, and val](https://huggingface.co/datasets/cais/mmlu) (MIT).

In a future release, we will provide this data with tooling to introspect the text content of the training set and the calibration set, relative to the estimated probabilities.

> [!TIP]
> The estimator has not been trained or calibrated against the held-out test sets of any LLM benchmarks. As such, you can assess new generative models in the way they will be used: Run the new models against the benchmark, and where appropriate update the SDM estimator with any relevant validation/dev data, and then send the generated output to Claude 3.7 with the Reexpress tool prompt. The preferred model(s) are those that maximize the proportion of index-conditional calibrated documents (i.e., in this case, those with a verified probability >= 0.95).
