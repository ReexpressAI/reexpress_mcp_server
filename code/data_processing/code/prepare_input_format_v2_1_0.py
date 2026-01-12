# Copyright Reexpress AI, Inc. All rights reserved.

from typing import Any, Callable, List, Tuple
import json
import argparse
import time
import codecs
import os
import random

from datasets import load_from_disk

import data_utils

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

GPT_5_DATA_KEY = "gpt-5-2025-08-07"
GPT_5_EXPLANATION_XML_TAG = "model1_explanation"
GEMINI_EXPLANATION_XML_TAG = "model2_explanation"

# In OpenVerification1:

# --field_selection="model3" \
# --field_selection_value="GEMINI_2_5_PRO_MODEL" \

# --field_selection="model4" \
# --field_selection_value="gpt-5-2025-08-07" \

# --model_label="model5" \
# --model_data_key="gpt-5.2-2025-12-11" \

# --model_label="model6" \
# --model_data_key="gemini-3-pro-preview" \


def _construct_agreement_template(model_explanation_string, topic=None):
    if topic is not None:
        topic_string = f"<topic> {topic} </topic> "
    else:
        topic_string = ""
    formatted_output_string = f"{topic_string}Do the following model explanations agree that the response is correct? {model_explanation_string} Yes or No?"
    return formatted_output_string


def get_confidence_soft_one_hot_list(is_verified, verbalized_confidence):
    # assert 0.0 <= verbalized_confidence <= 1.0, verbalized_confidence
    if is_verified:
        return [0.0, float(verbalized_confidence)]
    else:
        return [float(verbalized_confidence), 0.0]


def construct_agreement_template(row, is_eval=False):
    document_id = row["id"]
    user_question = row["user_question"]
    ai_response = row["ai_response"]
    # These correspond to the attributes in version v1.1.0 and are not used here: attributes = row["attributes"]
    document = f'<question> {user_question} </question> <ai_response> {ai_response} </ai_response>'
    label = int(row['label'])
    summary_from_gpt5 = row['model4_short_summary_of_original_question_and_response'].strip()
    summary_from_gpt5_2 = row['model5_short_summary_of_original_question_and_response'].strip()
    assert label in [0, 1]

    attributes = [0.0, 0.0, 0.0, 0.0]  # [GPT-5 class 0; GPT-5 class 1; Gemini class 0; Gemini class 1]

    gemini2_5_model_explanation = row["model3_short_explanation_for_classification_confidence"]
    gpt5_model_explanation = row["model4_short_explanation_for_classification_confidence"]

    gemini3_model_explanation = row["model6_short_explanation_for_classification_confidence"]
    gpt5_2_model_explanation = row["model5_short_explanation_for_classification_confidence"]

    gemini2_5_present = gemini2_5_model_explanation != ""
    gpt5_present = gpt5_model_explanation != ""

    gemini3_present = gemini3_model_explanation != ""
    gpt5_2_present = gpt5_2_model_explanation != ""

    gpt5_and_gemini_present = (gemini2_5_present or gemini3_present) and (gpt5_present or gpt5_2_present)

    model_explanation_string = ""
    # models: [GPT-5] + [Gemini], with preference for gpt-5.2-2025-12-11 and gemini-3-pro-preview
    # attributes are: [GPT-5 class 0; GPT-5 class 1; Gemini class 0; Gemini class 1]; else [0.0, 0.0, 0.0, 0.0]
    model1_label = "unavailable"
    model2_label = "unavailable"
    if gpt5_and_gemini_present:
        if gpt5_2_present:
            current_xml_tag = GPT_5_EXPLANATION_XML_TAG
            model_explanation_string = model_explanation_string.strip() + " " + f"<{current_xml_tag}> {gpt5_2_model_explanation} </{current_xml_tag}>"
            gpt5_attributes = get_confidence_soft_one_hot_list(row['model5_verification_classification'],
                                                               row['model5_confidence_in_classification'])
            attributes[0:2] = gpt5_attributes
            model1_label = row['model5']
        else:
            assert gpt5_present
            current_xml_tag = GPT_5_EXPLANATION_XML_TAG
            model_explanation_string = model_explanation_string.strip() + " " + f"<{current_xml_tag}> {gpt5_model_explanation} </{current_xml_tag}>"
            gpt5_attributes = get_confidence_soft_one_hot_list(row['model4_verification_classification'],
                                                               row['model4_confidence_in_classification'])
            attributes[0:2] = gpt5_attributes
            model1_label = row['model4']
        if gemini3_present:
            current_xml_tag = GEMINI_EXPLANATION_XML_TAG
            model_explanation_string = model_explanation_string.strip() + " " + f"<{current_xml_tag}> {gemini3_model_explanation} </{current_xml_tag}>"
            gemini_attributes = get_confidence_soft_one_hot_list(row['model6_verification_classification'],
                                                                 row['model6_confidence_in_classification'])
            attributes[2:4] = gemini_attributes
            model2_label = row['model6']
        else:
            assert gemini2_5_present
            current_xml_tag = GEMINI_EXPLANATION_XML_TAG
            model_explanation_string = model_explanation_string.strip() + " " + f"<{current_xml_tag}> {gemini2_5_model_explanation} </{current_xml_tag}>"
            gemini_attributes = get_confidence_soft_one_hot_list(row['model3_verification_classification'],
                                                                 row['model3_confidence_in_classification'])
            attributes[2:4] = gemini_attributes
            model2_label = row['model3']

    if not is_eval and not gpt5_and_gemini_present:
        # We only see both GPT-5 and Gemini (barring an API error).
        # However, for eval, we do need to consider,
        # but set fields to defaults, as applicable. This accounts for rejections from the API, which we count
        # as wrong predictions for the purposes of held-out evaluation.
        return None, None

    topic = None
    if gpt5_2_present and summary_from_gpt5_2 != "":
        topic = summary_from_gpt5_2
    elif gpt5_present and summary_from_gpt5 != "":
        topic = summary_from_gpt5
    model_explanation_string = model_explanation_string.strip()

    agreement_prompt = _construct_agreement_template(model_explanation_string=model_explanation_string, topic=topic)
    new_dict = {}
    new_dict[REEXPRESS_LABEL_KEY] = label
    new_dict[REEXPRESS_ID_KEY] = document_id
    new_dict[REEXPRESS_DOCUMENT_KEY] = document
    new_dict["agreement_prompt"] = agreement_prompt
    new_dict["summary"] = topic if topic is not None else ""
    new_dict[REEXPRESS_ATTRIBUTES_KEY] = attributes
    # new_dict["user_question"] = user_question
    # new_dict["ai_response"] = ai_response
    new_dict["info"] = row["info"]
    new_dict["meta"] = row["meta"]
    new_dict["gpt5_present"] = gpt5_present
    new_dict["gemini2_5_present"] = gemini2_5_present
    new_dict["gpt5_2_present"] = gpt5_2_present
    new_dict["gemini3_present"] = gemini3_present
    # For convenience, we also store the model names:
    new_dict[GPT_5_EXPLANATION_XML_TAG] = model1_label
    new_dict[GEMINI_EXPLANATION_XML_TAG] = model2_label

    return new_dict, gpt5_and_gemini_present


def main():
    parser = argparse.ArgumentParser(description="-----Construct Reexpress format-----")
    parser.add_argument("--input_datasets_file", default="", help="")
    parser.add_argument("--output_train_dir", default="", help="")
    parser.add_argument("--output_validation_dir", default="", help="")
    parser.add_argument("--output_eval_dir", default="", help="")
    options = parser.parse_args()

    random.seed(42)
    start_time = time.time()
    ds = load_from_disk(options.input_datasets_file)
    train = []
    validation = []
    # first, process eval
    split_name = "eval"
    for field_selection_value in ds[split_name].unique('info'):
        filtered_dataset = ds[split_name].filter(lambda x: x['info'] == field_selection_value)
        output_file = os.path.join(options.output_eval_dir, f"{field_selection_value}.jsonl")
        for row in filtered_dataset:
            reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=True)
            assert reexpress_obj is not None
            data_utils.save_by_appending_json_lines(output_file, [reexpress_obj])
    # here, mmlu_validation is also held-out eval
    split_name = "validation"
    field_selection_value = "mmlu_validation"
    filtered_dataset = ds[split_name].filter(lambda x: x['info'] == field_selection_value)
    output_file = os.path.join(options.output_eval_dir, f"{field_selection_value}.jsonl")
    for row in filtered_dataset:
        reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=True)
        assert reexpress_obj is not None
        data_utils.save_by_appending_json_lines(output_file, [reexpress_obj])
    for field_selection_value in ['mmlu_pro_validation', 'hl-fever_dev', 'openthoughts', 'mmlu_dev']:
        filtered_dataset = ds[split_name].filter(lambda x: x['info'] == field_selection_value)
        for row in filtered_dataset:
            reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=False)
            if reexpress_obj is not None:
                assert gpt5_and_gemini_present
                validation.append(reexpress_obj)

    split_name = "train"
    filtered_dataset = ds[split_name].filter(lambda x: x['meta'] == 'openthoughts_v4_v5_v6_v7_v8_split1.train')
    print(f"openthoughts_v4_v5_v6_v7_v8_split1.train: {len(filtered_dataset)}")

    for row in filtered_dataset:
        reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=False)
        if reexpress_obj is not None:
            assert gpt5_and_gemini_present
            train.append(reexpress_obj)

    filtered_dataset = ds[split_name].filter(lambda x: x['meta'] != 'openthoughts_v4_v5_v6_v7_v8_split1.train')
    for row in filtered_dataset:
        reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=False)
        if reexpress_obj is not None:
            assert gpt5_and_gemini_present
            if row["info"] == 'challenge_examples_2025_07_08':  # ensure challenge appears in estimator ca/tr
                validation.append(reexpress_obj)
            else:
                train.append(reexpress_obj)

    print(f"validation: {len(validation)}")
    print(f"train: {len(train)}")

    output_file = os.path.join(options.output_validation_dir, f"calibration.jsonl")
    data_utils.save_json_lines(filename_with_path=output_file, json_list=validation)

    output_file = os.path.join(options.output_train_dir, f"train.jsonl")
    data_utils.save_json_lines(filename_with_path=output_file, json_list=train)

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()
