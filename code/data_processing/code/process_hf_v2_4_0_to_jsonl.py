# Copyright Reexpress AI, Inc. All rights reserved.

import argparse
import time
# import os
# import random

from datasets import load_from_disk, load_dataset

import data_utils

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"


def get_documents_from_openverification(attested_ids):
    uuid_to_documents = {}
    dataset = load_dataset("ReexpressAI/OpenVerification1")
    for split_name in ["eval", "validation", "train"]:
        for row in dataset[split_name]:
            if row[REEXPRESS_ID_KEY] in attested_ids:
                uuid_to_documents[row[REEXPRESS_ID_KEY]] = \
                    f"<question> {row['user_question']} </question> <ai_response> {row['ai_response']} </ai_response>"
    return uuid_to_documents


def get_ids(dataset):
    uuids_set = set()
    for split in dataset.keys():
        for row in dataset[split]:
            uuids_set.add(row[REEXPRESS_ID_KEY])
    print(f"Total ids: {len(uuids_set)}")
    return uuids_set


def get_jsonl_splits(dataset_split, uuid_to_documents):
    data = []
    for row in dataset_split:
        document_text = ""
        if uuid_to_documents is not None and row[REEXPRESS_ID_KEY] in uuid_to_documents:
            document_text = uuid_to_documents[row[REEXPRESS_ID_KEY]]
        data.append({REEXPRESS_ID_KEY: row[REEXPRESS_ID_KEY],
                     REEXPRESS_LABEL_KEY: row[REEXPRESS_LABEL_KEY],
                     REEXPRESS_DOCUMENT_KEY: document_text,
                     REEXPRESS_EMBEDDING_KEY: row[REEXPRESS_EMBEDDING_KEY]
                     })
    return data


def main():
    parser = argparse.ArgumentParser(description="-----Construct Reexpress JSONL format-----")
    parser.add_argument("--output_train_file", default="", help="")
    parser.add_argument("--output_calibration_file", default="", help="")
    parser.add_argument("--output_openthoughts_eval_file", default="", help="")
    parser.add_argument("--output_mathnet_eval_eval_file", default="", help="")
    parser.add_argument("--include_document_text_in_output_jsonl",
                        default=False, action='store_true',
                        help="This currently only considers the rows in OpenVerification1.")
    options = parser.parse_args()

    #random.seed(42)
    start_time = time.time()
    print(f'Loading ReexpressMCPServer_v2_4_0_data from "ReexpressAI/ReexpressMCPServer_v2_4_0_data"')
    mcp_input_dataset = load_dataset("ReexpressAI/ReexpressMCPServer_v2_4_0_data")
    if options.include_document_text_in_output_jsonl:
        attested_ids = get_ids(mcp_input_dataset)
        # modify as needed for other source datasets:
        uuid_to_documents = get_documents_from_openverification(attested_ids)
    else:
        uuid_to_documents = None

    data_utils.save_json_lines(filename_with_path=options.output_train_file,
                               json_list=get_jsonl_splits(
                                   mcp_input_dataset['openverification_train_and_adaptation_train'],
                                   uuid_to_documents))
    data_utils.save_json_lines(filename_with_path=options.output_calibration_file,
                               json_list=get_jsonl_splits(
                                   mcp_input_dataset['mathnet_train'],
                                   uuid_to_documents))
    data_utils.save_json_lines(filename_with_path=options.output_openthoughts_eval_file,
                               json_list=get_jsonl_splits(
                                   mcp_input_dataset['openthoughts_eval'],
                                   uuid_to_documents))
    data_utils.save_json_lines(filename_with_path=options.output_mathnet_eval_eval_file,
                               json_list=get_jsonl_splits(
                                   mcp_input_dataset['mathnet_eval'],
                                   uuid_to_documents))

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()
