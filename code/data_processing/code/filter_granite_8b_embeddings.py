# Copyright Reexpress AI, Inc. All rights reserved.

"""
In rare cases (e.g., 9 out of 229601 training instances in v2.1.0), Gemini-2.5-pro output extraneous characters
which exceeded the max length of the Granite 8b model. We remove those here. This script also collects summary
stats on character lengths of the input.
"""

import json
import argparse
import time
from pathlib import Path
import codecs
import numpy as np
import data_utils


VERIFICATION_CLASSIFICATION_KEY = "verification_classification"
CONFIDENCE_IN_CLASSIFICATION_KEY = "confidence_in_classification"
SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY = "short_explanation_for_classification_confidence"

FIELD_PROBABILITIES_KEY = "field_probabilities"
LOG_PROB_MODEL_RESPONSE_KEY = "LOG_PROB_MODEL"
REASONING_MODEL_RESPONSE_KEY = "REASONING_MODEL"
GEMINI_2_5_PRO_MODEL_RESPONSE_KEY = "GEMINI_2_5_PRO_MODEL"

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

EXPECTED_EMBEDDING_SIZE = 8194


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj["id"])
    return existing_ids


def filter_embedding_streaming(options):
    count_incomplete_responses = 0
    incomplete_responses_aggreement_prompt_length_characters = []
    complete_responses_aggreement_prompt_length_characters = []
    output_file = options.output_file
    if Path(output_file).exists():
        existing_ids = get_existing_ids(output_file)
    else:
        existing_ids = set()

    instance_i = -1
    with codecs.open(options.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            instance_i += 1
            if instance_i % 25000 == 0:
                print(f"Currently processing instance {instance_i}")
            if json_obj["id"] in existing_ids:
                continue
            if json_obj[REEXPRESS_EMBEDDING_KEY] == [0.0] * EXPECTED_EMBEDDING_SIZE:
                count_incomplete_responses += 1
                incomplete_responses_aggreement_prompt_length_characters.append(len(json_obj["agreement_prompt"]))
                continue
            else:
                complete_responses_aggreement_prompt_length_characters.append(len(json_obj["agreement_prompt"]))
            data_utils.save_by_appending_json_lines(output_file, [json_obj])
            existing_ids.add(json_obj["id"])
    print(f"Count of documents with embedding set to 0's: {count_incomplete_responses}")
    print(f"Among missing embeddings, agreement prompt (in characters): "
          f"mean: {np.mean(incomplete_responses_aggreement_prompt_length_characters)}, "
          f"min: {np.min(incomplete_responses_aggreement_prompt_length_characters)}, "
          f"max: {np.max(incomplete_responses_aggreement_prompt_length_characters)}")
    print(f"Among complete embeddings, agreement prompt (in characters): "
          f"mean: {np.mean(complete_responses_aggreement_prompt_length_characters)}, "
          f"min: {np.min(complete_responses_aggreement_prompt_length_characters)}, "
          f"max: {np.max(complete_responses_aggreement_prompt_length_characters)}")


def main():
    parser = argparse.ArgumentParser(description="-----[Add embedding data to JSON objects]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--output_file", default="", help="")

    options = parser.parse_args()

    start_time = time.time()
    filter_embedding_streaming(options)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()
