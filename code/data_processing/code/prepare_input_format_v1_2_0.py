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
GPT_4_1_EXPLANATION_XML_TAG = "model3_explanation"
O4_MINI_EXPLANATION_XML_TAG = "model4_explanation"
GPT_5_EXPLANATION_XML_TAG = "model1_explanation"
GEMINI_2_5_EXPLANATION_XML_TAG = "model2_explanation"


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


def construct_agreement_template(row, is_eval=False, only_gpt5=True):
    document_id = row["id"]
    user_question = row["user_question"]
    ai_response = row["ai_response"]
    # These correspond to the attributes from v1.1.0 and are not used here: attributes = row["attributes"]
    document = f'<question> {user_question} </question> <ai_response> {ai_response} </ai_response>'
    label = int(row['label'])
    summary_from_gpt5 = row['model4_short_summary_of_original_question_and_response'].strip()
    assert label in [0, 1]

    attributes = [0.0, 0.0, 0.0, 0.0]  # [GPT-5 class 0; GPT-5 class 1; Gemini class 0; Gemini class 1]

    gemini2point5_model_explanation = row["model3_short_explanation_for_classification_confidence"]
    gpt5_model_explanation = row["model4_short_explanation_for_classification_confidence"]
    gemini_present = gemini2point5_model_explanation != ""
    gpt5_present = gpt5_model_explanation != ""
    gpt5_and_gemini_present = gemini_present and gpt5_present
    gpt41_present = row["model1_short_explanation_for_classification_confidence"] != ""
    o4mini_present = row["model2_short_explanation_for_classification_confidence"] != ""

    model_explanation_string = ""
    # models: [GPT-4.1] + [o4-mini] + [GPT-5] + [Gemini];
    # At test, we only see [GPT-5] + [Gemini]. The other configurations are primarily intended for pretraining.
    # attributes are only ever [GPT-5 class 0; GPT-5 class 1; Gemini class 0; Gemini class 1]; else [0.0, 0.0, 0.0, 0.0]
    if gpt5_present:
        current_xml_tag = GPT_5_EXPLANATION_XML_TAG
        model_explanation_string = model_explanation_string.strip() + " " + f"<{current_xml_tag}> {gpt5_model_explanation} </{current_xml_tag}>"
        gpt5_attributes = get_confidence_soft_one_hot_list(row['model4_verification_classification'],
                                                           row['model4_confidence_in_classification'])
        attributes[0:2] = gpt5_attributes
    if gemini_present:
        current_xml_tag = GEMINI_2_5_EXPLANATION_XML_TAG
        model_explanation_string = model_explanation_string.strip() + " " + f"<{current_xml_tag}> {gemini2point5_model_explanation} </{current_xml_tag}>"
        gemini_attributes = get_confidence_soft_one_hot_list(row['model3_verification_classification'],
                                                             row['model3_confidence_in_classification'])
        attributes[2:4] = gemini_attributes

    prefix_model_explanation_string = ""
    if gpt41_present:
        current_xml_tag = GPT_4_1_EXPLANATION_XML_TAG
        model_explanation = row["model1_short_explanation_for_classification_confidence"]
        prefix_model_explanation_string = prefix_model_explanation_string.strip() + " " + f"<{current_xml_tag}> {model_explanation} </{current_xml_tag}>"
    if o4mini_present:
        current_xml_tag = O4_MINI_EXPLANATION_XML_TAG
        model_explanation = row["model2_short_explanation_for_classification_confidence"]
        prefix_model_explanation_string = prefix_model_explanation_string.strip() + " " + f"<{current_xml_tag}> {model_explanation} </{current_xml_tag}>"

    if only_gpt5 or gpt5_and_gemini_present:  # In this case, we only consider at most GPT-5 and Gemini
        if not is_eval and not gpt5_and_gemini_present:
            # If we only seek the instances with gpt-5, then we skip the instances without both GPT-5 and Gemini.
            # Why? Because at test, we only see both GPT-5 and Gemini (barring an API error).
            # However, for eval, we do need to consider,
            # but set fields to defaults, as applicable.
            return None, None
    else:  # In this case, we additionally consider the older models.
        if prefix_model_explanation_string != "":
            model_explanation_string = \
                f"{prefix_model_explanation_string.strip()} {model_explanation_string.strip()}".strip()

    topic = None
    if summary_from_gpt5 != "":
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
    new_dict["gemini_present"] = gemini_present
    return new_dict, gpt5_and_gemini_present


if __name__ == "__main__":
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
    pretrain = []
    pretrain_for_subsequent_support_addition = []
    validation = []
    # first, process eval
    split_name = "eval"
    for field_selection_value in ds[split_name].unique('info'):
        filtered_dataset = ds[split_name].filter(lambda x: x['info'] == field_selection_value)
        output_file = os.path.join(options.output_eval_dir, f"{field_selection_value}.jsonl")
        for row in filtered_dataset:
            reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=True, only_gpt5=True)
            if reexpress_obj is not None:
                data_utils.save_by_appending_json_lines(output_file, [reexpress_obj])
    # here, mmlu_validation is also held-out eval
    split_name = "validation"
    field_selection_value = "mmlu_validation"
    filtered_dataset = ds[split_name].filter(lambda x: x['info'] == field_selection_value)
    output_file = os.path.join(options.output_eval_dir, f"{field_selection_value}.jsonl")
    for row in filtered_dataset:
        reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=True, only_gpt5=True)
        if reexpress_obj is not None:
            data_utils.save_by_appending_json_lines(output_file, [reexpress_obj])
    for field_selection_value in ['mmlu_pro_validation', 'hl-fever_dev', 'openthoughts', 'mmlu_dev']:
        filtered_dataset = ds[split_name].filter(lambda x: x['info'] == field_selection_value)
        for row in filtered_dataset:
            reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=False, only_gpt5=False)
            if reexpress_obj is not None:
                if field_selection_value == 'mmlu_dev':  # mmlu dev goes to pretrain
                    pretrain.append(reexpress_obj)
                else:
                    if gpt5_and_gemini_present:
                        validation.append(reexpress_obj)
                    else:
                        pretrain.append(reexpress_obj)

    split_name = "train"
    filtered_dataset = ds[split_name].filter(lambda x: x['meta'] == 'openthoughts_v4_v5_v6_v7_v8_split1.train')
    print(f"openthoughts_v4_v5_v6_v7_v8_split1.train: {len(filtered_dataset)}")
    gpt_5__v4_v5_v6_v7_v8_split1 = []

    for row in filtered_dataset:
        reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=False, only_gpt5=False)
        if reexpress_obj is not None:
            if gpt5_and_gemini_present:
                gpt_5__v4_v5_v6_v7_v8_split1.append(reexpress_obj)
            else:
                pretrain.append(reexpress_obj)
    print(f"gpt_5__v4_v5_v6_v7_v8_split1: {len(gpt_5__v4_v5_v6_v7_v8_split1)}")
    random.shuffle(gpt_5__v4_v5_v6_v7_v8_split1)
    # half to pretrain:
    total_gpt_5__v4_v5_v6_v7_v8_split1 = len(gpt_5__v4_v5_v6_v7_v8_split1)
    split_size = total_gpt_5__v4_v5_v6_v7_v8_split1 // 2
    train.extend(gpt_5__v4_v5_v6_v7_v8_split1[0:split_size])
    pretrain.extend(gpt_5__v4_v5_v6_v7_v8_split1[split_size:])  # add some examples with gpt5 to pretraining
    pretrain_for_subsequent_support_addition.extend(gpt_5__v4_v5_v6_v7_v8_split1[split_size:])  # convenience
    filtered_dataset = ds[split_name].filter(lambda x: x['meta'] != 'openthoughts_v4_v5_v6_v7_v8_split1.train')
    for row in filtered_dataset:
        reexpress_obj, gpt5_and_gemini_present = construct_agreement_template(row, is_eval=False, only_gpt5=False)
        if reexpress_obj is not None:
            if gpt5_and_gemini_present:
                if row["info"] == 'challenge_examples_2025_07_08':  # ensure challenge appears in estimator ca/tr
                    validation.append(reexpress_obj)
                else:
                    train.append(reexpress_obj)
            else:
                pretrain.append(reexpress_obj)

    print(f"validation: {len(validation)}")
    print(f"train: {len(train)}")
    print(f"pretrain: {len(pretrain)}")
    print(f"pretrain subset with GPT-5 and Gemini: {len(pretrain_for_subsequent_support_addition)}")
    output_file = os.path.join(options.output_validation_dir, f"calibration.jsonl")
    data_utils.save_json_lines(filename_with_path=output_file, json_list=validation)

    output_file = os.path.join(options.output_train_dir, f"train.jsonl")
    data_utils.save_json_lines(filename_with_path=output_file, json_list=train)

    output_file = os.path.join(options.output_train_dir, f"pretrain.jsonl")
    data_utils.save_json_lines(filename_with_path=output_file, json_list=pretrain)
    output_file = os.path.join(options.output_train_dir, f"pretrain_gpt5_and_gemini.jsonl")
    data_utils.save_json_lines(filename_with_path=output_file, json_list=pretrain_for_subsequent_support_addition)

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
