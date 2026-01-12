# Copyright Reexpress AI, Inc. All rights reserved.

"""
This constructs the document database for an existing model. The text comes from a HuggingFace datasets dataset.

In this version, not all database instances are from the same version of the GPT and Gemini. For GPT, the training
examples are from GPT-5 or GPT-5.2, and for Gemini, the training examples are from Gemini-2.5 or Gemini-3.

"""

import torch
import argparse
import codecs
import json
import time

from datasets import load_from_disk
from datasets import load_dataset

import utils_model
import mcp_utils_sqlite_document_db_controller

GPT5_PRESENT = "gpt5_present"
GEMINI2_5_PRESENT = "gemini2_5_present"
GPT5_2_PRESENT = "gpt5_2_present"
GEMINI3_PRESENT = "gemini3_present"


def retrieve_model_documents_ids(options):
    model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"), load_for_inference=True)
    assert len(model.train_uuids) == len(set(model.train_uuids)), \
        "Unexpected Error: The training documents ids are not unique."
    print(f"Support set size: {len(model.train_uuids)}")
    return set(model.train_uuids)


def get_agreement_classification(options, train_uuids_set):
    # We back this out from the embedding which contains the agreement model's output logits as the final two indexes
    document_id_2_agreement_classification_int = {}
    document_id_2_label_int = {}
    document_id_2_model_identity_dict = {}
    with codecs.open(options.best_iteration_train_split_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            document_id = json_obj["id"]
            label = json_obj["label"]
            no_prob = json_obj["embedding"][-2]
            yes_prob = json_obj["embedding"][-1]
            agreement_classification_int = int(no_prob < yes_prob)
            assert document_id in train_uuids_set
            train_uuids_set.remove(document_id)
            document_id_2_agreement_classification_int[document_id] = agreement_classification_int
            document_id_2_label_int[document_id] = label
            document_id_2_model_identity_dict[document_id] = {
                GPT5_PRESENT: bool(json_obj[GPT5_PRESENT]),
                GEMINI2_5_PRESENT: bool(json_obj[GEMINI2_5_PRESENT]),
                GPT5_2_PRESENT: bool(json_obj[GPT5_2_PRESENT]),
                GEMINI3_PRESENT: bool(json_obj[GEMINI3_PRESENT])
            }

    assert len(train_uuids_set) == 0
    return document_id_2_agreement_classification_int, document_id_2_label_int, document_id_2_model_identity_dict


def get_row_by_id(dataset_dict, document_id_2_hf_idx, target_id):
    for split_name in dataset_dict:
        if target_id in document_id_2_hf_idx[split_name]:
            idx = document_id_2_hf_idx[split_name][target_id]
            return dataset_dict[split_name][idx], split_name
    return None, None


def init_db(database_file: str, document_id_2_agreement_classification_int, document_id_2_label_int,
            document_id_2_model_identity_dict,
            expected_support_size: int, dataset_dict, document_id_2_hf_idx, dataset_label: str):
    db = mcp_utils_sqlite_document_db_controller.DocumentDatabase(database_file)
    missing_documents = 0
    for document_id in document_id_2_agreement_classification_int:
        model1_label = "unavailable"
        model2_label = "unavailable"
        agreement_classification_int = document_id_2_agreement_classification_int[document_id]
        label_int = document_id_2_label_int[document_id]
        model_identity_dict = document_id_2_model_identity_dict[document_id]
        row, _ = get_row_by_id(dataset_dict, document_id_2_hf_idx, document_id)
        if row is None:
            missing_documents += 1
            # default values:
            success = db.add_document(
                document_id=document_id,
                model1_summary="",
                model1_explanation="",
                model2_explanation="",
                model3_explanation="",
                model4_explanation="",
                model1_classification_int=0,
                model2_classification_int=0,
                model3_classification_int=0,
                model4_classification_int=0,
                agreement_model_classification_int=agreement_classification_int,
                label_int=label_int,
                label_was_updated_int=0,
                document_source="N/A",
                info=f"{model1_label},{model2_label}",
                user_question="",
                ai_response=""
            )
        else:
            # Note the remapping of the model names.
            # Model indices are remapped to match the agreement model's expected ordering
            # as of v2.1.0 of the MCP Server.
            # Model 1: GPT-5.2 if available, else GPT-5
            # Model 2: Gemini-3-pro-preview if available, else Gemini-2.5-pro

            model1_summary = ""
            model1_explanation = ""
            model2_explanation = ""
            model1_classification_int = 0
            model2_classification_int = 0

            gemini2_5_model_label = "model3"
            gpt5_model_label = "model4"

            gemini3_model_label = "model6"
            gpt5_2_model_label = "model5"

            if model_identity_dict[GPT5_2_PRESENT]:
                model1_summary = row[f"{gpt5_2_model_label}_short_summary_of_original_question_and_response"]
                model1_explanation = row[f"{gpt5_2_model_label}_short_explanation_for_classification_confidence"]
                model1_classification_int = row[f'{gpt5_2_model_label}_verification_classification']
                model1_label = row[f"{gpt5_2_model_label}"]
            else:
                assert model_identity_dict[GPT5_PRESENT]
                model1_summary = row[f"{gpt5_model_label}_short_summary_of_original_question_and_response"]
                model1_explanation = row[f"{gpt5_model_label}_short_explanation_for_classification_confidence"]
                model1_classification_int = row[f'{gpt5_model_label}_verification_classification']
                model1_label = row[f"{gpt5_model_label}"]

            if model_identity_dict[GEMINI3_PRESENT]:
                model2_explanation = row[f"{gemini3_model_label}_short_explanation_for_classification_confidence"]
                model2_classification_int = row[f'{gemini3_model_label}_verification_classification']
                model2_label = row[f"{gemini3_model_label}"]
            else:
                assert model_identity_dict[GEMINI2_5_PRESENT]
                model2_explanation = row[f"{gemini2_5_model_label}_short_explanation_for_classification_confidence"]
                model2_classification_int = row[f'{gemini2_5_model_label}_verification_classification']
                # Here, we override to use a similar naming convention as the other models
                # model2_label = row[f"{GEMINI2_5_PRESENT}"]
                assert row[f"{gemini2_5_model_label}"] == "GEMINI_2_5_PRO_MODEL"
                model2_label = "gemini-2.5-pro"

            success = db.add_document(
                document_id=document_id,
                model1_summary=model1_summary,
                model1_explanation=model1_explanation,
                model2_explanation=model2_explanation,
                model3_explanation='',
                model4_explanation='',
                model1_classification_int=model1_classification_int,
                model2_classification_int=model2_classification_int,
                model3_classification_int=0,
                model4_classification_int=0,
                agreement_model_classification_int=agreement_classification_int,
                label_int=label_int,
                label_was_updated_int=0,
                document_source=f"{dataset_label}: {row['info']}",
                info=f"{model1_label},{model2_label}",
                user_question=row['user_question'],
                ai_response=row['ai_response']
            )
        assert success
    count = db.get_document_count()
    print(f"Database created with {count} entries.")
    if missing_documents > 0:
        print(f"There were {missing_documents} documents not present in the HF dataset. "
              f"They were added with empty strings for the model explanations and the model classifications were set "
              f"to False.")
    assert count == expected_support_size


def main():
    parser = argparse.ArgumentParser(description="-----[Construct database]-----")
    parser.add_argument("--model_dir", default="", help="model_dir")
    parser.add_argument("--best_iteration_train_split_file", default="", help="best_iteration_train_split")
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
    print(f"This assumes the agreement model's classification is encoded in the input embedding.")
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

    train_uuids_set = retrieve_model_documents_ids(options)
    expected_support_size = len(train_uuids_set)
    document_id_2_agreement_classification_int, document_id_2_label_int, document_id_2_model_identity_dict = \
        get_agreement_classification(options, train_uuids_set)
    init_db(database_file=options.database_file,
            document_id_2_agreement_classification_int=document_id_2_agreement_classification_int,
            document_id_2_label_int=document_id_2_label_int,
            document_id_2_model_identity_dict=document_id_2_model_identity_dict,
            expected_support_size=expected_support_size,
            dataset_dict=ds, document_id_2_hf_idx=document_id_2_hf_idx, dataset_label=options.dataset_label)

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()

