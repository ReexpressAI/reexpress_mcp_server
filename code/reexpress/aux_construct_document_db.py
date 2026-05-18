# Copyright Reexpress AI, Inc. All rights reserved.

"""
This constructs the document database for an existing model. The text comes from datasets hosted on HuggingFace.

This script is unique to the release version of the model. For example, this version does not preprocess a
classification from an agreement model, since the hidden-states/embedding model is "gemini-embedding-2", rather
than a local LM. Use the version in the GitHub release archive, as applicable.

In this version, text is optionally pulled from "ReexpressAI/OpenVerification1",
"ReexpressAI/OpenVerification1_aux_adaptation_examples", and/or "ReexpressAI/OpenVerification1_aux_mathnet".

"""

import torch
import argparse
import time
from pathlib import Path

from datasets import load_from_disk
from datasets import load_dataset

import utils_model
import mcp_utils_sqlite_document_db_controller
import constants

GPT_MODEL_LABEL_KEY = constants.MCP_SERVER_MODEL1_NAME
GEMINI_MODEL_LABEL_KEY = constants.MCP_SERVER_MODEL2_NAME


def retrieve_model_documents_ids(model_dir):
    model = utils_model.load_model_torch(model_dir, torch.device("cpu"), load_for_inference=True)
    assert len(model.train_uuids) == len(set(model.train_uuids)), \
        "Unexpected Error: The training documents ids are not unique."
    print(f"Support set size: {len(model.train_uuids)}")
    train_uuids = model.train_uuids
    train_labels = model.train_labels
    train_document_id_2_label = {}
    for document_id, label in zip(train_uuids, train_labels):
        train_document_id_2_label[document_id] = label.item()
    return train_document_id_2_label


def get_row_by_id(dataset_dict, document_id_2_hf_idx, target_id):
    for split_name in dataset_dict:
        if target_id in document_id_2_hf_idx[split_name]:
            idx = document_id_2_hf_idx[split_name][target_id]
            return dataset_dict[split_name][idx], split_name
    return None, None


def init_db(database_file: str):
    return mcp_utils_sqlite_document_db_controller.DocumentDatabase(database_file)


def add_dataset_to_db(db: mcp_utils_sqlite_document_db_controller.DocumentDatabase,
                      train_document_id_2_label: dict[str, int],
                      dataset_dict, document_id_2_hf_idx, dataset_label: str):

    for document_id in train_document_id_2_label:
        # Note the remapping of the model names.
        # Model indices are remapped to match the agreement model's expected ordering
        # as of v2.4.0 of the MCP Server.
        # Model 1: GPT-5.X
        # Model 2: Gemini-3.X-pro-preview
        agreement_classification_int = -1  # not used in this version
        agreement_representation_model_name = constants.MCP_SERVER_API_EMBEDDING_MODEL_NAME
        label_int = train_document_id_2_label[document_id]
        row, _ = get_row_by_id(dataset_dict, document_id_2_hf_idx, document_id)
        if row is None:
            continue
        # The field names in the datasets differ, so each is processed separately below:
        if dataset_label == "OpenVerification1":
            model1_string_indicator_name = 'model9'
            model1_label = row[model1_string_indicator_name].strip()
            assert model1_label == GPT_MODEL_LABEL_KEY
            model2_string_indicator_name = 'model8'
            model2_label = row[model2_string_indicator_name].strip()
            assert model2_label == GEMINI_MODEL_LABEL_KEY

            model1_summary = \
                row[f"{model1_string_indicator_name}_short_summary_of_original_question_and_response"]
            model1_explanation = \
                row[f"{model1_string_indicator_name}_short_explanation_for_classification_confidence"]
            model1_classification_int = \
                row[f'{model1_string_indicator_name}_verification_classification']

            model2_explanation = \
                row[f"{model2_string_indicator_name}_short_explanation_for_classification_confidence"]
            model2_classification_int = \
                row[f'{model2_string_indicator_name}_verification_classification']

            document_source = f"{dataset_label}: {row['info']}"
            user_question = row['user_question']
            ai_response = row['ai_response']
        elif dataset_label == "OpenVerification1_aux_mathnet":
            model1_label = GPT_MODEL_LABEL_KEY
            model2_label = GEMINI_MODEL_LABEL_KEY
            assert label_int == int(row["v1_verification_is_for_original_solution"]), \
                "In this version, the label is expected to match that of the dataset row."
            user_question = row["problem_markdown"]
            if label_int == 1:
                # Note that solutions_markdown is a list; chosen_solution_index determines the index to use
                ai_response = row["solutions_markdown"][row["chosen_solution_index"]]
                field_prefix = "original_solution_verification"
            elif label_int == 0:
                ai_response = row["synthetic_negative_gpt-5.5-2026-04-23"]
                field_prefix = "synthetic_negative_verification"
            else:
                raise ValueError(f"Unexpected label: {label_int}")
            model1_summary = \
                row[f'{field_prefix}_{GPT_MODEL_LABEL_KEY}_short_summary_of_original_question_and_response']
            model1_explanation = \
                row[f"{field_prefix}_{GPT_MODEL_LABEL_KEY}_short_explanation_for_classification_confidence"]
            model1_classification_int = \
                row[f'{field_prefix}_{GPT_MODEL_LABEL_KEY}_verification_classification']

            model2_explanation = \
                row[f"{field_prefix}_{GEMINI_MODEL_LABEL_KEY}_short_explanation_for_classification_confidence"]
            model2_classification_int = \
                row[f'{field_prefix}_{GEMINI_MODEL_LABEL_KEY}_verification_classification']

            document_source = f"{dataset_label}"
        elif dataset_label == "OpenVerification1_aux_adaptation_examples":
            assert label_int == int(row["label"]), \
                "In this version, the label is expected to match that of the dataset row."
            model1_label = row["model1_name"].strip()
            assert model1_label == GPT_MODEL_LABEL_KEY
            model2_label = row["model2_name"].strip()
            assert model2_label == GEMINI_MODEL_LABEL_KEY

            model1_summary = \
                row["model1_summary"]
            model1_explanation = \
                row["model1_explanation"]
            model1_classification_int = \
                row["model1_classification"]

            model2_explanation = \
                row["model2_explanation"]
            model2_classification_int = \
                row["model2_classification"]

            document_source = f"{dataset_label}: attached_documents: " \
                              f"{'Yes' if len(row['attached_file_names']) > 0 else 'No'}"
            user_question = row['question']
            ai_response = row['ai_response']
        else:
            raise ValueError(f"Unexpected dataset label: {dataset_label}")

        success = db.add_document(
            document_id=document_id,
            model1_summary=model1_summary,
            model1_explanation=model1_explanation,
            model2_explanation=model2_explanation,
            model3_explanation='',
            model4_explanation='',
            model1_classification_int=int(model1_classification_int),
            model2_classification_int=int(model2_classification_int),
            model3_classification_int=0,
            model4_classification_int=0,
            agreement_model_classification_int=int(agreement_classification_int),
            label_int=int(label_int),
            label_was_updated_int=0,
            document_source=document_source,
            info=f"{model1_label},{model2_label},{agreement_representation_model_name}",
            user_question=user_question,
            ai_response=ai_response
        )
        assert success
    count = db.get_document_count()
    print(f"Database now contains {count} entries.")


def process_one_hf_dataset(db, train_document_id_2_label,
                           load_hf_dataset_from_disk, input_datasets_file, dataset_label):
    if load_hf_dataset_from_disk:
        ds = load_from_disk(input_datasets_file)
        print(f"Successfully loaded {input_datasets_file} from disk.")
    else:
        ds = load_dataset(input_datasets_file)
        print(f"Successfully loaded {input_datasets_file}, which is the current version from HF Hub.")

    document_id_2_hf_idx = {}
    for split_name in ds:
        split_ids = ds[split_name]['id']
        document_id_2_hf_idx[split_name] = {id_val: idx for idx, id_val in enumerate(split_ids)}

    add_dataset_to_db(db=db,
                      train_document_id_2_label=train_document_id_2_label,
                      dataset_dict=ds, document_id_2_hf_idx=document_id_2_hf_idx, dataset_label=dataset_label)


def main():
    parser = argparse.ArgumentParser(description="-----[Construct database]-----")
    parser.add_argument("--model_dir", default="", help="model_dir")
    parser.add_argument("--database_file", default="", help="database_file")
    parser.add_argument("--hf_open_verification_datasets_file", default="",
                        help="ReexpressAI/OpenVerification1, "
                             "or an archived on-disk version if --load_hf_open_verification_from_disk. "
                             "If not provided, the text for these instances will not be added to the database.")
    parser.add_argument("--load_hf_open_verification_from_disk", default=False, action='store_true',
                        help="If provided, then --hf_open_verification_datasets_file should be a path to "
                             "a locally saved datasets file.")
    parser.add_argument("--hf_mathnet_datasets_file", default="",
                        help="ReexpressAI/OpenVerification1_aux_mathnet, "
                             "or an archived on-disk version if --load_hf_mathnet_from_disk. "
                             "If not provided, the text for these instances will not be added to the database.")
    parser.add_argument("--load_hf_mathnet_from_disk", default=False, action='store_true',
                        help="If provided, then --hf_mathnet_datasets_file should be a path to "
                             "a locally saved datasets file.")
    parser.add_argument("--hf_adaptation_datasets_file", default="",
                        help="ReexpressAI/OpenVerification1_aux_adaptation_examples, "
                             "or an archived on-disk version if --load_hf_adaptation_from_disk. "
                             "If not provided, the text for these instances will not be added to the database.")
    parser.add_argument("--load_hf_adaptation_from_disk", default=False, action='store_true',
                        help="If provided, then --hf_adaptation_datasets_file should be a path to "
                             "a locally saved datasets file.")

    options = parser.parse_args()

    start_time = time.time()

    if Path(options.database_file).exists():
        print(f"{options.database_file} exists. The existing database will be extended.")
    else:
        print(f"{options.database_file} does not exist. A new database will be created.")

    db = init_db(database_file=options.database_file)

    train_document_id_2_label = retrieve_model_documents_ids(model_dir=options.model_dir)
    total_support_size = len(train_document_id_2_label)

    if options.hf_open_verification_datasets_file.strip() != "":
        process_one_hf_dataset(db,
                               train_document_id_2_label=train_document_id_2_label,
                               load_hf_dataset_from_disk=options.load_hf_open_verification_from_disk,
                               input_datasets_file=options.hf_open_verification_datasets_file,
                               dataset_label="OpenVerification1")
    else:
        print(f"Skipping ReexpressAI/OpenVerification1")

    if options.hf_mathnet_datasets_file.strip() != "":
        process_one_hf_dataset(db,
                               train_document_id_2_label=train_document_id_2_label,
                               load_hf_dataset_from_disk=options.load_hf_mathnet_from_disk,
                               input_datasets_file=options.hf_mathnet_datasets_file,
                               dataset_label="OpenVerification1_aux_mathnet")
    else:
        print(f"Skipping ReexpressAI/OpenVerification1_aux_mathnet")

    if options.hf_adaptation_datasets_file.strip() != "":
        process_one_hf_dataset(db,
                               train_document_id_2_label=train_document_id_2_label,
                               load_hf_dataset_from_disk=options.load_hf_adaptation_from_disk,
                               input_datasets_file=options.hf_adaptation_datasets_file,
                               dataset_label="OpenVerification1_aux_adaptation_examples")
    else:
        print(f"Skipping ReexpressAI/OpenVerification1_aux_adaptation_examples")

    count = db.get_document_count()
    database_coverage = total_support_size-count
    if database_coverage < 0:
        raise ValueError(f"ERROR: The database size exceeds the model's support set size by {abs(database_coverage)} "
                         f"document(s). An existing database file from a different model "
                         f"may have been provided as --database_file.")
    print(f"Database now contains {count} entries for a support size of {total_support_size} instances. "
          f"{database_coverage} document(s) lack a database entry.")

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()
