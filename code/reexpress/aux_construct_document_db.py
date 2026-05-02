# Copyright Reexpress AI, Inc. All rights reserved.

"""
This constructs the document database for an existing model. The text comes from a HuggingFace datasets dataset.

This script is unique to the release version of the model. For example, this version does not preprocess a
classification from an agreement model, since the hidden-states/embedding model is "gemini-embedding-2", rather
than a local LM. Use the version in the GitHub release archive, as applicable.

"""

import torch
import argparse
import time

from datasets import load_from_disk
from datasets import load_dataset

import utils_model
import mcp_utils_sqlite_document_db_controller

GPT_MODEL_LABEL_KEY = "gpt-5.4-2026-03-05"
GEMINI_MODEL_LABEL_KEY = "gemini-3.1-pro-preview"


def retrieve_model_documents_ids(options):
    model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"), load_for_inference=True)
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


def init_db(database_file: str, train_document_id_2_label: dict[str, int],
            expected_support_size: int, dataset_dict, document_id_2_hf_idx, dataset_label: str):
    db = mcp_utils_sqlite_document_db_controller.DocumentDatabase(database_file)
    missing_documents = 0
    for document_id in train_document_id_2_label:
        # Note the remapping of the model names.
        # Model indices are remapped to match the agreement model's expected ordering
        # as of v2.3.0 of the MCP Server.
        # Model 1: GPT-5.X
        # Model 2: Gemini-3.X-pro-preview
        model1_label = "unavailable"
        model2_label = "unavailable"
        agreement_classification_int = -1  # not used in this version
        model1_summary = ""
        model1_explanation = ""
        model2_explanation = ""
        model1_classification_int = 0
        model2_classification_int = 0
        document_source = "N/A"
        user_question = ""
        ai_response = ""
        label_int = train_document_id_2_label[document_id]
        row, _ = get_row_by_id(dataset_dict, document_id_2_hf_idx, document_id)
        if row is None:
            missing_documents += 1
        else:
            model1_string_indicator_name = 'model7'
            model1_label = row[model1_string_indicator_name].strip()
            assert model1_label == GPT_MODEL_LABEL_KEY
            model2_string_indicator_name = 'model8'
            model2_label = row[model2_string_indicator_name].strip()
            assert model2_label == GEMINI_MODEL_LABEL_KEY

            model1_summary = row[f"{model1_string_indicator_name}_short_summary_of_original_question_and_response"]
            model1_explanation = row[f"{model1_string_indicator_name}_short_explanation_for_classification_confidence"]
            model1_classification_int = row[f'{model1_string_indicator_name}_verification_classification']

            model2_explanation = row[f"{model2_string_indicator_name}_short_explanation_for_classification_confidence"]
            model2_classification_int = row[f'{model2_string_indicator_name}_verification_classification']

            document_source = f"{dataset_label}: {row['info']}"
            user_question = row['user_question']
            ai_response = row['ai_response']

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
            info=f"{model1_label},{model2_label}",
            user_question=user_question,
            ai_response=ai_response
        )
        assert success
    count = db.get_document_count()
    print(f"Database created with {count} entries.")
    if missing_documents > 0:
        print(f"There were {missing_documents} documents not present in the HF dataset. "
              f"They were added with empty strings for the model explanations and the model classifications were set "
              f"to False (class 0).")
    assert count == expected_support_size


def main():
    parser = argparse.ArgumentParser(description="-----[Construct database]-----")
    parser.add_argument("--model_dir", default="", help="model_dir")
    parser.add_argument("--database_file", default="", help="database_file")
    parser.add_argument("--load_hf_dataset_from_disk", default=False, action='store_true',
                        help="load_hf_dataset_from_disk")
    parser.add_argument("--input_datasets_file", default="ReexpressAI/OpenVerification1",
                        help="If --load_hf_dataset_from_disk, then --input_datasets_file should be a path to "
                             "a datasets file. Otherwise, --input_datasets_file should be the name of the applicable "
                             "dataset on the HuggingFace Hub.")
    parser.add_argument("--dataset_label", default="OpenVerification1",
                        help="This is saved to the database's document_source field as: "
                             "dataset_label: [additional info]")

    options = parser.parse_args()

    start_time = time.time()
    if options.load_hf_dataset_from_disk:
        ds = load_from_disk(options.input_datasets_file)
        print(f"Successfully loaded {options.input_datasets_file} from disk.")
    else:
        ds = load_dataset(options.input_datasets_file)
        print(f"Successfully loaded {options.input_datasets_file}, which is the current version from HF Hub.")

    document_id_2_hf_idx = {}
    for split_name in ds:
        split_ids = ds[split_name]['id']
        document_id_2_hf_idx[split_name] = {id_val: idx for idx, id_val in enumerate(split_ids)}

    train_document_id_2_label = retrieve_model_documents_ids(options)
    expected_support_size = len(train_document_id_2_label)
    init_db(database_file=options.database_file,
            train_document_id_2_label=train_document_id_2_label,
            expected_support_size=expected_support_size,
            dataset_dict=ds, document_id_2_hf_idx=document_id_2_hf_idx, dataset_label=options.dataset_label)

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()

