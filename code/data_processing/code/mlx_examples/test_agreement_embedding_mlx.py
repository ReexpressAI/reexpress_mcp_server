# Copyright Reexpress AI, Inc. All rights reserved.

# Test harness: run get_local_embedding_for_agreement_prompt() over a JSON lines file with the MLX module.
# Each line must have "lm_model_summary" and "lm_model_explanation" string fields.
#
#   python test_agreement_embedding_mlx.py examples.jsonl --output out.jsonl --check_tokens
#
# Fixed to mlx-community/gemma-4-31b-it-4bit (see the module header). Embeddings are 3 x 5376 = 16128 dims
# (plus 2 with --include_yes_no_logits) and are not comparable to those from the bf16 PyTorch module.
# Peak MLX memory is printed at the end so the activation cost at --max_token_length can be read off directly.

import argparse
import json
import logging
import time

import mlx.core as mx

import mcp_utils_llm_api_gemma_4_31b_it_mlx as agreement


def _peak_memory_gb() -> float:
    getter = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory  # older MLX kept it under mx.metal
    return getter() / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--model_path", default=agreement.DEFAULT_MODEL_PATH)
    parser.add_argument("--max_token_length", type=int, default=8192)
    parser.add_argument("--no_max_pool", action="store_true")
    parser.add_argument("--include_yes_no_logits", action="store_true",
                        help="append the [No, Yes] logits to the embedding (off by default, matching the module's "
                             "convention; get_agreement_model_embedding currently rejects it)")
    parser.add_argument("--output", default=None, help="optional JSONL to write embeddings and classifications")
    parser.add_argument("--check_tokens", action="store_true",
                        help="for the first example, print how the prompt ends, the top next-token candidates, and "
                             "a short greedy continuation (Yes/No token, thinking-block, and decoding sanity check)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input_jsonl) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    print(f"{len(examples)} examples; model {args.model_path}")

    if args.check_tokens and examples:
        ex = examples[0]
        agreement.check_label_tokens(
            agreement.get_model_explanations_formatted_as_binary_agreement_prompt(
                ex["lm_model_summary"], ex["lm_model_explanation"]),
            model_path=args.model_path)

    out = open(args.output, "w") if args.output else None
    for i, ex in enumerate(examples):
        start = time.perf_counter()
        embedding, classification = agreement.get_local_embedding_for_agreement_prompt(
            ex["lm_model_summary"], ex["lm_model_explanation"],
            model_path=args.model_path, max_token_length=args.max_token_length,
            include_max_pool=not args.no_max_pool, include_yes_no_logits=args.include_yes_no_logits,
        )
        elapsed = time.perf_counter() - start
        if embedding is None:
            print(f"[{i}] FAILED ({elapsed:.2f}s)")
            continue
        print(f"[{i}] agree={classification} dim={len(embedding)} ({elapsed:.2f}s)")
        if out:
            out.write(json.dumps({**ex, "agreement_classification": classification, "embedding": embedding}) + "\n")
    if out:
        out.close()
    print(f"peak MLX memory: {_peak_memory_gb():.1f} GB")


if __name__ == "__main__":
    main()
