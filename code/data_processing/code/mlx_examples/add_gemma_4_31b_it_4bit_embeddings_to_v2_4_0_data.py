# Copyright Reexpress AI, Inc. All rights reserved.

# Fixed to mlx-community/gemma-4-31b-it-4bit (see the module header). Embeddings are 3 x 5376 = 16128 dims.
# max-pool over the sequence :: mean-pool over the sequence :: hidden-state of the final token (that estimates Yes | No)

import json
import sys
import numpy as np
import logging
import argparse
import time
from pathlib import Path
import codecs

import data_utils
import mcp_utils_llm_api_gemma_4_31b_it_mlx as agreement

logger = logging.getLogger(__name__)


REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

EXPECTED_EMBEDDING_SIZE = 16128  # 3 x 5376 = 16128


def get_classification_signed_indicator_list(is_verified):
    if is_verified:
        return [-1.0, 1.0]
    else:
        return [1.0, -1.0]


def print_summary(header_label, list_to_process, total=None):
    if total is not None and total > 0:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)} "
            f"\t({(len(list_to_process)/total) * 100}%) of {total}")
    else:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)}")


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj[REEXPRESS_ID_KEY])
    return existing_ids


def construct_embedding_streaming(options):
    # count_incomplete_responses = 0
    output_file = options.output_file
    if Path(output_file).exists():
        existing_ids = get_existing_ids(output_file)
    else:
        existing_ids = set()

    acc = []
    acc_by_class = {}
    acc_by_predicted_class = {}
    for class_i in range(options.class_size):
        acc_by_class[class_i] = []
        acc_by_predicted_class[class_i] = []

    instance_i = -1
    with codecs.open(options.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            instance_i += 1
            if instance_i % 500 == 0:
                print(f"Currently processing instance {instance_i}")

            output_id = json_obj[REEXPRESS_ID_KEY]
            if output_id in existing_ids:
                continue

            embedding, llm_classification, truncated_prompt = \
                agreement.get_local_embedding_for_agreement_prompt_with_prompt(
                    json_obj["model1_summary"], json_obj["model1_explanation"],
                    model_path=agreement.DEFAULT_MODEL_PATH, max_token_length=options.max_token_length,
                    include_max_pool=not options.no_max_pool, include_yes_no_logits=False,
                )

            if embedding is None:
                print(f"LINE_{instance_i}: {json_obj[REEXPRESS_ID_KEY]}")
                # In principle, this case should never occur with this model, so we exit to investigate further.
                sys.exit(1)
            assert len(embedding) == EXPECTED_EMBEDDING_SIZE, \
                f"Unexpected embedding size {len(embedding)} (expected {EXPECTED_EMBEDDING_SIZE})"
            llm_classification = int(llm_classification)  # module returns a bool; store as 0/1 like the label

            json_obj[REEXPRESS_EMBEDDING_KEY] = embedding
            json_obj[REEXPRESS_ATTRIBUTES_KEY] = \
                get_classification_signed_indicator_list(json_obj["model1_classification_int"])
            json_obj["prompt"] = truncated_prompt  # non-truncated prompt in json_obj["agreement_prompt"]
            json_obj["llm_classification"] = llm_classification  # for reference
            acc.append(llm_classification == json_obj[REEXPRESS_LABEL_KEY])
            acc_by_class[json_obj[REEXPRESS_LABEL_KEY]].append(
                llm_classification == json_obj[REEXPRESS_LABEL_KEY])
            acc_by_predicted_class[llm_classification].append(
                llm_classification == json_obj[REEXPRESS_LABEL_KEY])
            # Note there is additional reference metadata from the original json_obj
            data_utils.save_by_appending_json_lines(output_file, [json_obj])
            existing_ids.add(output_id)

    # print(f"Count of documents with embedding set to 0's: {count_incomplete_responses}")
    model_label = agreement.DEFAULT_MODEL_PATH
    print_summary(f"{model_label} accuracy", acc, total=len(acc))
    print(f"Class-conditional accuracy (i.e., stratified by TRUE class):")
    for class_i in range(options.class_size):
        print_summary(f"{model_label} accuracy true class {class_i}",
                      acc_by_class[class_i], total=len(acc))
    print(f"Prediction-conditional accuracy (i.e., stratified by PREDICTED class):")
    for class_i in range(options.class_size):
        print_summary(f"{model_label} accuracy predicted class {class_i}",
                      acc_by_predicted_class[class_i], total=len(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Add embedding data to JSON objects]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--class_size", default=2, type=int, help="class_size")
    parser.add_argument("--max_token_length", default=8192, type=int, help="max_token_length")
    parser.add_argument("--no_max_pool", action="store_true")
    parser.add_argument("--output_file", default="", help="")

    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    start_time = time.time()
    print(f"Using a max_token_length of {options.max_token_length}")
    assert not options.no_max_pool  # currently max-pool is expected
    print(f"Concat max-pool: {not options.no_max_pool}")
    construct_embedding_streaming(options)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
