# Copyright Reexpress AI, Inc. All rights reserved.

# Preprocessing of the v2.4.0 text for analysis, here focusing on model 1 ("gpt-5.5-2026-04-23").
# The embedding has been removed and should be re-added using the
# representation model used for calibration.

import argparse
import time

from datasets import load_from_disk, load_dataset

import data_utils

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

MCP_SERVER_MODEL1_NAME = "gpt-5.5-2026-04-23"

def get_model_explanations_formatted_as_binary_agreement_prompt(lm_model_summary,
                                                                lm_model_explanation) -> str:
    # "mlx-community/gemma-4-31b-it-4bit" prompt
    # This constructs an un-truncated prompt for reference. The script that generates the embeddings has an
    # adjustable max-length limit.
    formatted_output_string = f"<topic> {lm_model_summary} </topic> Does the following model explanation agree that the response is correct? <model_explanation> {lm_model_explanation} </model_explanation> Answer with a single word: Yes or No?"
    return formatted_output_string


def get_documents_from_openverification(attested_ids, dataset_label: str):
    uuid_to_documents = {}
    dataset = load_dataset("ReexpressAI/OpenVerification1")
    model1_missing_count = 0
    for split_name in ["eval", "validation", "train"]:
        for row in dataset[split_name]:
            if row[REEXPRESS_ID_KEY] in attested_ids:
                model1_string_indicator_name = 'model9'
                model1_label = row[model1_string_indicator_name].strip()
                if model1_label != MCP_SERVER_MODEL1_NAME:
                    print(f"{model1_label} != {MCP_SERVER_MODEL1_NAME}. row: {row}")
                    model1_missing_count += 1
                else:
                    assert len(row[f"{model1_string_indicator_name}_short_summary_of_original_question_and_response"]) > 0 and \
                           len(row[f"{model1_string_indicator_name}_short_explanation_for_classification_confidence"]) > 0
                agreement_prompt = get_model_explanations_formatted_as_binary_agreement_prompt(
                                    lm_model_summary=row[f"{model1_string_indicator_name}_short_summary_of_original_question_and_response"],
                                    lm_model_explanation=row[f"{model1_string_indicator_name}_short_explanation_for_classification_confidence"])
                qa_text = f"<question> {row['user_question']} </question> <ai_response> {row['ai_response']} </ai_response>"
                document_source = f"{dataset_label}: {row['info']}"
                json_obj = {"qa": qa_text,
                            "model1_summary": row[f"{model1_string_indicator_name}_short_summary_of_original_question_and_response"],
                            "model1_explanation": row[f"{model1_string_indicator_name}_short_explanation_for_classification_confidence"],
                            "model1_classification_int": row[f'{model1_string_indicator_name}_verification_classification'],
                            "agreement_prompt": agreement_prompt,
                            REEXPRESS_DOCUMENT_KEY: f"{qa_text} <agreement_prompt> {agreement_prompt} </agreement_prompt> <lm_self_verification_classification> {row[f'{model1_string_indicator_name}_verification_classification']} </lm_self_verification_classification>",  # convenience for reading in Reexpress two
                            "document_source": document_source
                }
                uuid_to_documents[row[REEXPRESS_ID_KEY]] = json_obj
    print(f"Count of rows missing model 1: {model1_missing_count}")
    return uuid_to_documents


def get_documents_from_OpenVerification1_aux_adaptation_examples(attested_ids, uuid_to_documents, dataset_label: str,
                                                                 uuids_to_labels=None):
    assert len(uuid_to_documents) > 0 and uuids_to_labels is not None
    dataset = load_dataset("ReexpressAI/OpenVerification1_aux_adaptation_examples")
    for split_name in ["v2.4.0"]:
        for row in dataset[split_name]:
            if row[REEXPRESS_ID_KEY] in attested_ids and row[REEXPRESS_ID_KEY] in uuids_to_labels:
                label_int = uuids_to_labels[row[REEXPRESS_ID_KEY]]
                assert label_int == int(row["label"]), \
                    "In this version, the label is expected to match that of the dataset row."
                model1_label = row["model1_name"].strip()
                assert model1_label == MCP_SERVER_MODEL1_NAME
                model1_summary = \
                    row["model1_summary"]
                model1_explanation = \
                    row["model1_explanation"]
                model1_classification_int = \
                    row["model1_classification"]

                document_source = f"{dataset_label}: attached_documents: " \
                                  f"{'Yes' if len(row['attached_file_names']) > 0 else 'No'}"
                user_question = row['question']
                ai_response = row['ai_response']

                assert len(model1_summary) > 0 and \
                       len(model1_explanation) > 0
                agreement_prompt = get_model_explanations_formatted_as_binary_agreement_prompt(
                                    lm_model_summary=model1_summary,
                                    lm_model_explanation=model1_explanation)
                qa_text = f"<question> {user_question} </question> <ai_response> {ai_response} </ai_response>"
                json_obj = {"qa": qa_text,
                            "model1_summary": model1_summary,
                            "model1_explanation": model1_explanation,
                            "model1_classification_int": model1_classification_int,
                            "agreement_prompt": agreement_prompt,
                            REEXPRESS_DOCUMENT_KEY: f"{qa_text} <agreement_prompt> {agreement_prompt} </agreement_prompt> <lm_self_verification_classification> {model1_classification_int} </lm_self_verification_classification>",  # convenience for reading in Reexpress two
                            "document_source": document_source
                }
                uuid_to_documents[row[REEXPRESS_ID_KEY]] = json_obj
    return uuid_to_documents


def get_documents_from_mathnet(attested_ids, dataset_label: str, uuids_to_labels=None):
    # with MathNet, we also need labels from uuids_to_labels, because these determine whether the example is using
    # the original with a correct solution (label 1) or a synthetic negative (label 0), because each row in the
    # HF dataset provides both. v1_verification_is_for_original_solution contains this, but we verify with the
    # original labels in case the choice changes in the future.
    assert uuids_to_labels is not None and len(uuids_to_labels) > 0
    uuid_to_documents = {}
    dataset = load_dataset("ReexpressAI/OpenVerification1_aux_mathnet")
    for split_name in ["train", "eval"]:
        for row in dataset[split_name]:
            if row[REEXPRESS_ID_KEY] in attested_ids and row[REEXPRESS_ID_KEY] in uuids_to_labels:
                label_int = uuids_to_labels[row[REEXPRESS_ID_KEY]]
                model1_label = MCP_SERVER_MODEL1_NAME
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
                    row[f'{field_prefix}_{model1_label}_short_summary_of_original_question_and_response']
                model1_explanation = \
                    row[f"{field_prefix}_{model1_label}_short_explanation_for_classification_confidence"]
                model1_classification_int = \
                    row[f'{field_prefix}_{model1_label}_verification_classification']
                document_source = f"{dataset_label}"

                assert len(model1_summary) > 0 and \
                       len(model1_explanation) > 0
                agreement_prompt = get_model_explanations_formatted_as_binary_agreement_prompt(
                                    lm_model_summary=model1_summary,
                                    lm_model_explanation=model1_explanation)
                qa_text = f"<question> {user_question} </question> <ai_response> {ai_response} </ai_response>"
                json_obj = {"qa": qa_text,
                            "model1_summary": model1_summary,
                            "model1_explanation": model1_explanation,
                            "model1_classification_int": model1_classification_int,
                            "agreement_prompt": agreement_prompt,
                            REEXPRESS_DOCUMENT_KEY: f"{qa_text} <agreement_prompt> {agreement_prompt} </agreement_prompt> <lm_self_verification_classification> {model1_classification_int} </lm_self_verification_classification>",  # convenience for reading in Reexpress two
                            "document_source": document_source
                }
                uuid_to_documents[row[REEXPRESS_ID_KEY]] = json_obj
    return uuid_to_documents


def get_documents_from_non_public_hle_file(input_json_list):
    # This for processing our held-out split of the HLE benchmark. This is not publicly available to avoid
    # contaminating the test set, but it can be freely recreated from the gated HLE dataset on
    # HuggingFace.
    json_list = []
    for row in input_json_list:
        assert row['model1_name'] == MCP_SERVER_MODEL1_NAME
        assert row['mcp_server_version'] == '2.4.0'

        model1_summary = \
            row["model1_summary"]
        model1_explanation = \
            row["model1_explanation"]
        model1_classification_int = \
            row["model1_classification"]

        document_source = f"non-public-hle-mc-text: submitted_time: " \
                          f"{row['submitted_time']}"
        user_question = row['question']
        ai_response = row['ai_response']

        assert len(model1_summary) > 0 and \
               len(model1_explanation) > 0
        agreement_prompt = get_model_explanations_formatted_as_binary_agreement_prompt(
                            lm_model_summary=model1_summary,
                            lm_model_explanation=model1_explanation)
        qa_text = f"<question> {user_question} </question> <ai_response> {ai_response} </ai_response>"
        json_obj = {REEXPRESS_ID_KEY: row[REEXPRESS_ID_KEY],
                    REEXPRESS_LABEL_KEY: row[REEXPRESS_LABEL_KEY],
                    REEXPRESS_EMBEDDING_KEY: [],  # existing embedding is ignored; will be overwritten
                    "qa": qa_text,
                    "model1_summary": model1_summary,
                    "model1_explanation": model1_explanation,
                    "model1_classification_int": model1_classification_int,
                    "agreement_prompt": agreement_prompt,
                    REEXPRESS_DOCUMENT_KEY: f"{qa_text} <agreement_prompt> {agreement_prompt} </agreement_prompt> <lm_self_verification_classification> {model1_classification_int} </lm_self_verification_classification>",  # convenience for reading in Reexpress two
                    "document_source": document_source
                    }
        json_list.append(json_obj)
    return json_list


def get_ids(dataset):
    uuids_set = set()
    uuids_to_labels = {}
    for split in dataset.keys():
        for row in dataset[split]:
            uuids_set.add(row[REEXPRESS_ID_KEY])
            uuids_to_labels[row[REEXPRESS_ID_KEY]] = row[REEXPRESS_LABEL_KEY]

    print(f"Total ids: {len(uuids_set)}")
    assert len(uuids_set) == len(uuids_to_labels)
    return uuids_set, uuids_to_labels


def get_jsonl_splits(dataset_split, uuid_to_documents):
    data = []
    for row in dataset_split:
        json_obj = uuid_to_documents[row[REEXPRESS_ID_KEY]]
        data.append({REEXPRESS_ID_KEY: row[REEXPRESS_ID_KEY],
                     REEXPRESS_LABEL_KEY: row[REEXPRESS_LABEL_KEY],
                     REEXPRESS_DOCUMENT_KEY: json_obj[REEXPRESS_DOCUMENT_KEY],
                     REEXPRESS_EMBEDDING_KEY: [],  # existing embedding is ignored; will be overwritten
                     # Some duplication here for convenience to avoid reparsing:
                     "qa": json_obj["qa"],
                     "model1_summary": json_obj["model1_summary"],
                     "model1_explanation": json_obj["model1_explanation"],
                     "model1_classification_int": json_obj["model1_classification_int"],
                     "agreement_prompt": json_obj["agreement_prompt"],
                     "document_source": json_obj["document_source"]
                     })
    return data


def main():
    parser = argparse.ArgumentParser(description="-----Construct Reexpress JSONL format-----")
    parser.add_argument("--output_train_file", default="", help="")
    parser.add_argument("--output_calibration_file", default="", help="")
    parser.add_argument("--output_openthoughts_eval_file", default="", help="")
    parser.add_argument("--output_mathnet_eval_eval_file", default="", help="")
    # for internal use:
    parser.add_argument("--input_non_public_hle_eval_file", default="",
                        help="internal use; should be blank to process public data")
    parser.add_argument("--output_non_public_hle_eval_file", default="",
                        help="internal use; should be blank to process public data")

    options = parser.parse_args()

    start_time = time.time()

    print(f'Loading ReexpressMCPServer_v2_4_0_data from "ReexpressAI/ReexpressMCPServer_v2_4_0_data"')
    mcp_input_dataset = load_dataset("ReexpressAI/ReexpressMCPServer_v2_4_0_data")
    attested_ids, attested_ids2labels = get_ids(mcp_input_dataset)

    # modify as needed for other source datasets:
    uuid_to_documents = get_documents_from_openverification(attested_ids,
                                                            dataset_label="OpenVerification1")
    uuid_to_documents = \
        get_documents_from_OpenVerification1_aux_adaptation_examples(
            attested_ids,
            uuid_to_documents,
            dataset_label="OpenVerification1_aux_adaptation_examples",
            uuids_to_labels=attested_ids2labels)
    data_utils.save_json_lines(filename_with_path=options.output_train_file,
                               json_list=get_jsonl_splits(
                                   mcp_input_dataset['openverification_train_and_adaptation_train'],
                                   uuid_to_documents))

    uuid_to_documents_mathnet = get_documents_from_mathnet(attested_ids,
                                                           dataset_label="OpenVerification1_aux_mathnet",
                                                           uuids_to_labels=attested_ids2labels)
    data_utils.save_json_lines(filename_with_path=options.output_calibration_file,
                               json_list=get_jsonl_splits(
                                   mcp_input_dataset['mathnet_train'],
                                   uuid_to_documents_mathnet))

    data_utils.save_json_lines(filename_with_path=options.output_openthoughts_eval_file,
                               json_list=get_jsonl_splits(
                                   mcp_input_dataset['openthoughts_eval'],
                                   uuid_to_documents))
    data_utils.save_json_lines(filename_with_path=options.output_mathnet_eval_eval_file,
                               json_list=get_jsonl_splits(
                                   mcp_input_dataset['mathnet_eval'],
                                   uuid_to_documents_mathnet))

    if options.input_non_public_hle_eval_file != "" and options.output_non_public_hle_eval_file != "":
        print(f"Processing final held-out (non-public) HLE dataset.")
        data_utils.save_json_lines(filename_with_path=options.output_non_public_hle_eval_file,
                                   json_list=get_documents_from_non_public_hle_file(
                                       data_utils.read_jsons_lines_file(options.input_non_public_hle_eval_file)))
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()
