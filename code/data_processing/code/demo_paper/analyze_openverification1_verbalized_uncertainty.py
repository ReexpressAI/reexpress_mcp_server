# Copyright Reexpress AI, Inc. All rights reserved.

import argparse
import random

from datasets import load_dataset
import numpy as np

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
# REEXPRESS_EMBEDDING_KEY = "embedding"

# GPT_5_DATA_KEY = "gpt-5-2025-08-07"
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

GPT_5_2_LABEL_KEY = "gpt-5.2-2025-12-11"
GEMINI_3_PRO_LABEL_KEY = "gemini-3-pro-preview"
JOINT_MODEL_LABEL_KEY = "gpt-5.2-2025-12-11_and_gemini-3-pro-preview"

MARGINAL_ACC_KEY = "m"
MARGINAL_ADMITTED_KEY = "m_prop"
CLASS_CONDITIONAL_ACC_KEY = "c"
CLASS_CONDITIONAL_ADMITTED_KEY = "c_prop"
PREDICTION_CONDITIONAL_ACC_KEY = "p"
PREDICTION_CONDITIONAL_ADMITTED_KEY = "p_prop"


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
    # These correspond to the attributes from v1.1.0 and are not used here: attributes = row["attributes"]
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
    gpt5_2_prediction = -1
    gpt5_2_confidence = -1
    gemini3_prediction = -1
    gemini3_confidence = -1
    if gpt5_and_gemini_present:
        if gpt5_2_present:
            current_xml_tag = GPT_5_EXPLANATION_XML_TAG
            model_explanation_string = model_explanation_string.strip() + " " + f"<{current_xml_tag}> {gpt5_2_model_explanation} </{current_xml_tag}>"
            gpt5_attributes = get_confidence_soft_one_hot_list(row['model5_verification_classification'],
                                                               row['model5_confidence_in_classification'])
            gpt5_2_prediction = int(row['model5_verification_classification'])
            gpt5_2_confidence = float(row['model5_confidence_in_classification'])
            attributes[0:2] = gpt5_attributes
            model1_label = row['model5']
        else:  # for reference, we consider the older model, but this is not used in the downstream analysis
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
            gemini3_prediction = int(row['model6_verification_classification'])
            gemini3_confidence = float(row['model6_confidence_in_classification'])
            attributes[2:4] = gemini_attributes
            model2_label = row['model6']
        else:  # for reference, we consider the older model, but this is not used in the downstream analysis
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
        return None, None, None, None, None, None, None

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

    return new_dict, gpt5_and_gemini_present, \
        gpt5_2_prediction, gpt5_2_confidence, gemini3_prediction, gemini3_confidence, gpt5_2_present and gemini3_present


def get_acc_prop_tuple(list_to_process, total=None):
    if total is not None and total > 0:
        prop = len(list_to_process)/total
    else:
        prop = 0.0
    acc = np.mean(list_to_process) if len(list_to_process) > 0 else 0.0
    return acc, prop


def print_summary(header_label, list_to_process, total=None):
    if total is not None and total > 0:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)} "
            f"\t({len(list_to_process)/total}) of {total}")
    else:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)}")


def get_decorated_string(float_proportion, display_string, decorate=False, decorate_alpha=0.95,
                         is_fully_rejected=False):
    if not decorate:
        return display_string
    if is_fully_rejected:
        return rf"\colorbox{{correctPredictionColor}}{{{display_string}}}"
    if float_proportion >= decorate_alpha:
        return rf"\colorbox{{correctPredictionColor}}{{{display_string}}}"
    else:
        return rf"\colorbox{{wrongPredictionColor}}{{{display_string}}}"


def get_float_as_display_significant_digits_string(float_proportion, decorate=False, decorate_alpha=0.95,
                                                   is_fully_rejected=False) -> str:
    if is_fully_rejected:
        return get_decorated_string(float_proportion, r"\allRejected", decorate=decorate,
                                    decorate_alpha=decorate_alpha, is_fully_rejected=is_fully_rejected)
    if float_proportion == 0.0:
        return get_decorated_string(float_proportion, "0.", decorate=decorate, decorate_alpha=decorate_alpha)
    elif float_proportion == 1.0:
        return get_decorated_string(float_proportion, "1.", decorate=decorate, decorate_alpha=decorate_alpha)
    if float_proportion < 0.005 and float_proportion != 0.0:
        return get_decorated_string(float_proportion, "<0.01", decorate=decorate, decorate_alpha=decorate_alpha)
    else:
        return get_decorated_string(float_proportion, f"{float_proportion:.2f}",
                                    decorate=decorate, decorate_alpha=decorate_alpha)


def get_latex_row(dataset_name, model_name, alpha, latex_rows_dict, estimator_label, numberOfClasses):
    running_latex_rows = []
    for class_label in range(numberOfClasses):
        conditional_acc = \
            get_float_as_display_significant_digits_string(
                latex_rows_dict[f"{CLASS_CONDITIONAL_ACC_KEY}{class_label}"],
                decorate=True, decorate_alpha=alpha,
                is_fully_rejected=latex_rows_dict[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{class_label}"] == 0.0)
        running_latex_rows.append(conditional_acc)
        admitted_proportion = \
            get_float_as_display_significant_digits_string(
                latex_rows_dict[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{class_label}"],
                decorate=False, decorate_alpha=alpha, is_fully_rejected=False)
        running_latex_rows.append(admitted_proportion)
    for class_label in range(numberOfClasses):
        conditional_acc = \
            get_float_as_display_significant_digits_string(
                latex_rows_dict[f"{PREDICTION_CONDITIONAL_ACC_KEY}{class_label}"],
                decorate=True, decorate_alpha=alpha,
                is_fully_rejected=latex_rows_dict[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{class_label}"] == 0.0)
        running_latex_rows.append(conditional_acc)
        admitted_proportion = \
            get_float_as_display_significant_digits_string(
                latex_rows_dict[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{class_label}"],
                decorate=False, decorate_alpha=alpha, is_fully_rejected=False)
        running_latex_rows.append(admitted_proportion)
    marginal_acc = get_float_as_display_significant_digits_string(
        latex_rows_dict[MARGINAL_ACC_KEY], decorate=True, decorate_alpha=alpha,
        is_fully_rejected=latex_rows_dict[MARGINAL_ADMITTED_KEY] == 0.0)
    running_latex_rows.append(marginal_acc)
    marginal_admitted_proportion = get_float_as_display_significant_digits_string(
        latex_rows_dict[MARGINAL_ADMITTED_KEY], decorate=False,
        decorate_alpha=alpha, is_fully_rejected=False)
    running_latex_rows.append(marginal_admitted_proportion)
    return " & ".join([dataset_name, model_name, estimator_label]) + " & " + " & ".join(running_latex_rows) + r"\\"


def init_latex_rows_dict(numberOfClasses):
    latex_rows_dict = {}
    latex_rows_dict[MARGINAL_ACC_KEY] = 0.0  # marginal accuracy
    latex_rows_dict[MARGINAL_ADMITTED_KEY] = 0.0  # marginal |Admitted| / |N|
    for class_label in range(numberOfClasses):
        latex_rows_dict[f"{CLASS_CONDITIONAL_ACC_KEY}{class_label}"] = 0.0  # class-conditional accuracy
        latex_rows_dict[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{class_label}"] = 0.0  # class-conditional |Admitted| / |N|
        latex_rows_dict[f"{PREDICTION_CONDITIONAL_ACC_KEY}{class_label}"] = 0.0  # prediction-conditional accuracy
        latex_rows_dict[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{class_label}"] = 0.0  # prediction-conditional |Admitted| / |N|
    return latex_rows_dict


def print_latex_row(dataset_name, number_of_classes, alpha,
                    latex_rows_dict_no_reject_gpt_5_2, latex_rows_dict_verbalized_uncertainty_at_alpha_gpt_5_2,
                    latex_rows_dict_no_reject_gemini3, latex_rows_dict_verbalized_uncertainty_at_alpha_gemini3,
                    latex_rows_dict_no_reject_joint, latex_rows_dict_verbalized_uncertainty_at_alpha_joint):

    print(f"Latex-formatted results table rows (alpha={alpha})")
    print(
        get_latex_row(dataset_name, r'$\modelGPTFiveTwo$', alpha,
                      latex_rows_dict_no_reject_gpt_5_2,
                      r'$\estimatorNoReject$', number_of_classes)
    )
    print(
        get_latex_row(dataset_name, r'$\modelGPTFiveTwo$', alpha,
                      latex_rows_dict_verbalized_uncertainty_at_alpha_gpt_5_2,
                      r'$\estimatorVerbProb$', number_of_classes)
    )
    print(
        get_latex_row(dataset_name, r'$\modelGeminiThree$', alpha,
                      latex_rows_dict_no_reject_gemini3,
                      r'$\estimatorNoReject$', number_of_classes)
    )
    print(
        get_latex_row(dataset_name, r'$\modelGeminiThree$', alpha,
                      latex_rows_dict_verbalized_uncertainty_at_alpha_gemini3,
                      r'$\estimatorVerbProb$', number_of_classes)
    )
    print(
        get_latex_row(dataset_name, r'$\modelGPTFiveTwoANDGeminiThreeAgreement$', alpha,
                      latex_rows_dict_no_reject_joint,
                      r'$\estimatorAgreeOrReject$', number_of_classes)
    )
    print(
        get_latex_row(dataset_name, r'$\modelGPTFiveTwoANDGeminiThreeAgreement$', alpha,
                      latex_rows_dict_verbalized_uncertainty_at_alpha_joint,
                      r'$\estimatorVerbProb$', number_of_classes)
    )


def calculate_summary_stats(model_label, number_of_classes, test_set_size, alpha,
                            class_conditional_accuracy, prediction_conditional_accuracy,
                            class_conditional_accuracy_at_alpha, prediction_conditional_accuracy_at_alpha,
                            marginal, marginal_at_alpha):
    latex_rows_dict_no_reject = init_latex_rows_dict(numberOfClasses=number_of_classes)
    latex_rows_dict_at_alpha = init_latex_rows_dict(numberOfClasses=number_of_classes)
    print(f"######## {model_label}: Conditional estimates ########")
    for label in range(number_of_classes):
        print(f"Label {label} ---")
        print_summary(f"Class-conditional accuracy: Label {label}",
                      class_conditional_accuracy[label], total=test_set_size)
        latex_rows_dict_no_reject[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_no_reject[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy[label], total=test_set_size)

        print_summary(f"\t>>Class-conditional >= {alpha} accuracy: \t\tLabel {label}",
                      class_conditional_accuracy_at_alpha[label], total=test_set_size)
        latex_rows_dict_at_alpha[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_at_alpha[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy_at_alpha[label], total=test_set_size)

        print_summary(f"Prediction-conditional accuracy: Label {label}",
                      prediction_conditional_accuracy[label], total=test_set_size)
        latex_rows_dict_no_reject[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_no_reject[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy[label], total=test_set_size)

        print_summary(f"\t>>Prediction-conditional >= {alpha} accuracy: "
                      f"\t\tLabel {label}",
                      prediction_conditional_accuracy_at_alpha[label], total=test_set_size)
        latex_rows_dict_at_alpha[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_at_alpha[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy_at_alpha[label], total=test_set_size)

    print(f"######## Marginal estimates ########")
    print(f"Marginal accuracy {model_label}: "
          f"{np.mean(marginal) if len(marginal) > 0 else 0.0} out of {len(marginal)}")
    latex_rows_dict_no_reject[MARGINAL_ACC_KEY], \
        latex_rows_dict_no_reject[MARGINAL_ADMITTED_KEY] = \
        get_acc_prop_tuple(marginal, total=test_set_size)

    # Marginal at alpha
    print(f"\t >= {alpha} Marginal accuracy {model_label}: "
          f"{np.mean(marginal_at_alpha) if len(marginal_at_alpha) > 0 else 0.0} out of {len(marginal_at_alpha)}")
    latex_rows_dict_at_alpha[MARGINAL_ACC_KEY], \
        latex_rows_dict_at_alpha[MARGINAL_ADMITTED_KEY] = \
        get_acc_prop_tuple(marginal_at_alpha, total=test_set_size)

    return latex_rows_dict_no_reject, latex_rows_dict_at_alpha


def process_data_split(ds, split_name, field_selection_value, number_of_classes, alpha):

    class_conditional = {}
    prediction_conditional = {}
    marginal = {}
    class_conditional_at_alpha = {}
    prediction_conditional_at_alpha = {}
    marginal_at_alpha = {}
    for model_class in [GPT_5_2_LABEL_KEY, GEMINI_3_PRO_LABEL_KEY, JOINT_MODEL_LABEL_KEY]:
        class_conditional_at_alpha[model_class] = {}
        prediction_conditional_at_alpha[model_class] = {}
        marginal_at_alpha[model_class] = []
        class_conditional[model_class] = {}
        prediction_conditional[model_class] = {}
        marginal[model_class] = []
        for c in range(number_of_classes):
            class_conditional_at_alpha[model_class][c] = []
            prediction_conditional_at_alpha[model_class][c] = []
            class_conditional[model_class][c] = []
            prediction_conditional[model_class][c] = []

    filtered_dataset = ds[split_name].filter(lambda x: x['info'] == field_selection_value)
    for row in filtered_dataset:
        reexpress_obj, _, \
            gpt5_2_prediction, gpt5_2_confidence, \
            gemini3_prediction, gemini3_confidence, \
            gpt5_2__and_gemini3_present = construct_agreement_template(row, is_eval=False)

        if gpt5_2__and_gemini3_present is not None and gpt5_2__and_gemini3_present:
            true_label = reexpress_obj[REEXPRESS_LABEL_KEY]
            # JOINT
            if gpt5_2_prediction == gemini3_prediction:
                class_conditional[JOINT_MODEL_LABEL_KEY][true_label].append(
                    true_label == gpt5_2_prediction)
                prediction_conditional[JOINT_MODEL_LABEL_KEY][gpt5_2_prediction].append(
                    true_label == gpt5_2_prediction)
                marginal[JOINT_MODEL_LABEL_KEY].append(
                    true_label == gpt5_2_prediction)
            if gpt5_2_confidence >= alpha and gemini3_confidence >= alpha and gpt5_2_prediction == gemini3_prediction:
                class_conditional_at_alpha[JOINT_MODEL_LABEL_KEY][true_label].append(
                    true_label == gpt5_2_prediction)
                prediction_conditional_at_alpha[JOINT_MODEL_LABEL_KEY][gpt5_2_prediction].append(
                    true_label == gpt5_2_prediction)
                marginal_at_alpha[JOINT_MODEL_LABEL_KEY].append(
                    true_label == gpt5_2_prediction)
            # GPT 5.2
            class_conditional[GPT_5_2_LABEL_KEY][true_label].append(
                true_label == gpt5_2_prediction)
            prediction_conditional[GPT_5_2_LABEL_KEY][gpt5_2_prediction].append(
                true_label == gpt5_2_prediction)
            marginal[GPT_5_2_LABEL_KEY].append(
                true_label == gpt5_2_prediction)
            if gpt5_2_confidence >= alpha:
                class_conditional_at_alpha[GPT_5_2_LABEL_KEY][true_label].append(
                    true_label == gpt5_2_prediction)
                prediction_conditional_at_alpha[GPT_5_2_LABEL_KEY][gpt5_2_prediction].append(
                    true_label == gpt5_2_prediction)
                marginal_at_alpha[GPT_5_2_LABEL_KEY].append(
                    true_label == gpt5_2_prediction)
            # Gemini-3-pro
            class_conditional[GEMINI_3_PRO_LABEL_KEY][true_label].append(
                true_label == gemini3_prediction)
            prediction_conditional[GEMINI_3_PRO_LABEL_KEY][gemini3_prediction].append(
                true_label == gemini3_prediction)
            marginal[GEMINI_3_PRO_LABEL_KEY].append(
                true_label == gemini3_prediction)
            if gemini3_confidence >= alpha:
                class_conditional_at_alpha[GEMINI_3_PRO_LABEL_KEY][true_label].append(
                    true_label == gemini3_prediction)
                prediction_conditional_at_alpha[GEMINI_3_PRO_LABEL_KEY][gemini3_prediction].append(
                    true_label == gemini3_prediction)
                marginal_at_alpha[GEMINI_3_PRO_LABEL_KEY].append(
                    true_label == gemini3_prediction)

    test_set_size = len(marginal[GPT_5_2_LABEL_KEY])
    assert test_set_size == len(marginal[GEMINI_3_PRO_LABEL_KEY])
    assert test_set_size >= len(marginal[JOINT_MODEL_LABEL_KEY])
    latex_rows_dict_no_reject_gpt_5_2, latex_rows_dict_verbalized_uncertainty_at_alpha_gpt_5_2 = \
        calculate_summary_stats(model_label=GPT_5_2_LABEL_KEY, number_of_classes=number_of_classes,
                                test_set_size=test_set_size, alpha=alpha,
                                class_conditional_accuracy=class_conditional[GPT_5_2_LABEL_KEY],
                                prediction_conditional_accuracy=prediction_conditional[GPT_5_2_LABEL_KEY],
                                class_conditional_accuracy_at_alpha=class_conditional_at_alpha[GPT_5_2_LABEL_KEY],
                                prediction_conditional_accuracy_at_alpha=prediction_conditional_at_alpha[GPT_5_2_LABEL_KEY],
                                marginal=marginal[GPT_5_2_LABEL_KEY],
                                marginal_at_alpha=marginal_at_alpha[GPT_5_2_LABEL_KEY])

    latex_rows_dict_no_reject_gemini3, latex_rows_dict_verbalized_uncertainty_at_alpha_gemini3 = \
        calculate_summary_stats(model_label=GEMINI_3_PRO_LABEL_KEY, number_of_classes=number_of_classes,
                                test_set_size=test_set_size, alpha=alpha,
                                class_conditional_accuracy=class_conditional[GEMINI_3_PRO_LABEL_KEY],
                                prediction_conditional_accuracy=prediction_conditional[GEMINI_3_PRO_LABEL_KEY],
                                class_conditional_accuracy_at_alpha=class_conditional_at_alpha[GEMINI_3_PRO_LABEL_KEY],
                                prediction_conditional_accuracy_at_alpha=prediction_conditional_at_alpha[GEMINI_3_PRO_LABEL_KEY],
                                marginal=marginal[GEMINI_3_PRO_LABEL_KEY],
                                marginal_at_alpha=marginal_at_alpha[GEMINI_3_PRO_LABEL_KEY])

    latex_rows_dict_no_reject_joint, latex_rows_dict_verbalized_uncertainty_at_alpha_joint = \
        calculate_summary_stats(model_label=JOINT_MODEL_LABEL_KEY, number_of_classes=number_of_classes,
                                test_set_size=test_set_size, alpha=alpha,
                                class_conditional_accuracy=class_conditional[JOINT_MODEL_LABEL_KEY],
                                prediction_conditional_accuracy=prediction_conditional[JOINT_MODEL_LABEL_KEY],
                                class_conditional_accuracy_at_alpha=class_conditional_at_alpha[JOINT_MODEL_LABEL_KEY],
                                prediction_conditional_accuracy_at_alpha=prediction_conditional_at_alpha[JOINT_MODEL_LABEL_KEY],
                                marginal=marginal[JOINT_MODEL_LABEL_KEY],
                                marginal_at_alpha=marginal_at_alpha[JOINT_MODEL_LABEL_KEY])

    if split_name == "validation" and field_selection_value == "mmlu_validation":
        dataset_name = r'$\datasetMMLUValidationBinaryVerification$'
    elif split_name == "eval" and field_selection_value == "openthoughts":
        dataset_name = r'$\OpenThoughtsTest$'
    else:
        assert False

    print_latex_row(dataset_name, number_of_classes, alpha,
                    latex_rows_dict_no_reject_gpt_5_2, latex_rows_dict_verbalized_uncertainty_at_alpha_gpt_5_2,
                    latex_rows_dict_no_reject_gemini3, latex_rows_dict_verbalized_uncertainty_at_alpha_gemini3,
                    latex_rows_dict_no_reject_joint, latex_rows_dict_verbalized_uncertainty_at_alpha_joint)


def main():
    parser = argparse.ArgumentParser(description="-----Evaluate verbalized uncertainty-----")
    parser.add_argument("--alpha", default=0.9, type=float, help="")
    parser.add_argument("--number_of_classes", default=2, type=int, help="")
    options = parser.parse_args()

    random.seed(42)

    alpha = options.alpha
    number_of_classes = options.number_of_classes
    print(f"Evaluated at alpha={alpha} for {number_of_classes} classes")

    ds = load_dataset("ReexpressAI/OpenVerification1")

    print("This evaluates the verbalized uncertainty for examples in OpenVerification1 with "
          "responses from GPT-5.2 and Gemini-3-pro-preview.")
    print(f"##################################")
    print(f"######## Dataset : MMLU-validation ########")
    print(f"################")
    process_data_split(ds=ds, split_name="validation", field_selection_value="mmlu_validation",
                       number_of_classes=number_of_classes, alpha=alpha)
    print(f"##################################")
    print(f"######## Dataset : OpenThoughts ########")
    print(f"################")
    process_data_split(ds=ds, split_name="eval", field_selection_value="openthoughts",
                       number_of_classes=number_of_classes, alpha=alpha)


if __name__ == "__main__":
    main()
