# Copyright Reexpress AI, Inc. All rights reserved.

import json
import argparse
import time
import codecs

import data_utils


kSLEEP_CONSTANT = 40
VERIFICATION_CLASSIFICATION_KEY = "verification_classification"
CONFIDENCE_IN_CLASSIFICATION_KEY = "confidence_in_classification"
SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY = "short_explanation_for_classification_confidence"

FIELD_PROBABILITIES_KEY = "field_probabilities"
LOG_PROB_MODEL_RESPONSE_KEY = "LOG_PROB_MODEL"
REASONING_MODEL_RESPONSE_KEY = "REASONING_MODEL"
# GEMINI_2_5_PRO_MODEL_RESPONSE_KEY = "GEMINI_2_5_PRO_MODEL"

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

# GPT5_MODEL = "gpt-5"
GPT_5_EXPLANATION_XML_TAG = "model1_explanation"
GEMINI_3_EXPLANATION_XML_TAG = "model2_explanation"

GEMINI_3_PRO_MODEL = "gemini-3-pro-preview"
GPT5_2_MODEL = "gpt-5.2-2025-12-11"


def construct_embedding_streaming(options):
    output_file = options.output_file
    retained_lines = 0
    total_lines = 0
    with codecs.open(options.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)

            if json_obj["gpt5_2_present"] and json_obj["gemini3_present"]:
                retained_lines += 1
                data_utils.save_by_appending_json_lines(output_file, [json_obj])
            total_lines += 1

    print(f"Number of instances with {GPT5_2_MODEL} and {GEMINI_3_PRO_MODEL}: {retained_lines}")
    print(f"Total lines: {total_lines}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Filter data by model availability]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--output_file", default="", help="")
    options = parser.parse_args()

    start_time = time.time()
    construct_embedding_streaming(options)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
