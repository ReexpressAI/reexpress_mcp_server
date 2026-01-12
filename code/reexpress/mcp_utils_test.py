# Copyright Reexpress AI, Inc. All rights reserved.

# test-time predictions and formatting for MCP server

import torch
import numpy as np

import constants


def _format_probability_as_string_percentage(valid_probability_float: float) -> str:
    threshold_as_string = (
        constants.floatProbToDisplaySignificantDigits(
            floatProb=valid_probability_float))
    return f"{threshold_as_string[2:]}%"


def get_formatted_sdm_estimator_output_string(verification_classification,
                                              calibration_reliability,
                                              gpt5_model_explanation,
                                              gemini_model_explanation,
                                              agreement_model_classification: bool,
                                              hr_class_conditional_accuracy: float) -> str:
    # If this changes, the docstring in reexpress_mcp_server.reexpress() should also be updated to avoid confusing
    # the downstream LLMs/agents.
    classification_confidence = \
        get_calibration_confidence_label(calibration_reliability=calibration_reliability,
                                         hr_class_conditional_accuracy=hr_class_conditional_accuracy)
    if agreement_model_classification:
        agreement_model_classification_string = "Yes"
    else:
        agreement_model_classification_string = "No"
    formatted_output_string = f"""
        <successfully_verified> {verification_classification} </successfully_verified> \n
        <confidence> {classification_confidence} </confidence> \n
        <model1_explanation> {gpt5_model_explanation} </model1_explanation> \n
        <model2_explanation> {gemini_model_explanation} </model2_explanation> \n
        <model3_agreement> {constants.AGREEMENT_MODEL_USER_FACING_PROMPT} {agreement_model_classification_string} </model3_agreement>
    """
    return formatted_output_string


def get_files_in_consideration_message(attached_files_names_list):
    if len(attached_files_names_list) > 0:
        files_in_consideration_message = f'The verification model had access to: ' \
                                         f'{",".join(attached_files_names_list)}\n\n'
    else:
        files_in_consideration_message = f'The verification model did not have access to any external files.\n\n'
    return files_in_consideration_message


def get_calibration_confidence_label(calibration_reliability: str, hr_class_conditional_accuracy: float,
                                     return_html_class=False) -> str:

    if calibration_reliability == constants.CALIBRATION_RELIABILITY_LABEL_OOD:
        classification_confidence_html_class = "negative"
        classification_confidence = "Out-of-distribution (unreliable)"
    elif calibration_reliability == constants.CALIBRATION_RELIABILITY_LABEL_HIGHEST:
        classification_confidence_html_class = "positive"
        classification_confidence = f">= {_format_probability_as_string_percentage(valid_probability_float=hr_class_conditional_accuracy)}"
    elif calibration_reliability == constants.CALIBRATION_RELIABILITY_LABEL_LOW__NEAR_CHANCE:
        classification_confidence_html_class = "near-random-chance"
        classification_confidence = f"< {_format_probability_as_string_percentage(valid_probability_float=constants.CALIBRATION_RELIABILITY_LABEL_LOW__NEAR_CHANCE_THRESHOLD)} (approaching random chance, so use with caution)"
    else:
        classification_confidence_html_class = "caution"
        # Switching to '<= 89%' (or equivalent relative to hr_class_conditional_accuracy with an offset of 0.01),
        # as some models may miss (or otherwise get confused by) the less than sign when
        # the output is '< 90%'.
        classification_confidence = f"<= {_format_probability_as_string_percentage(valid_probability_float=hr_class_conditional_accuracy-0.01)} (use with caution)"
    if return_html_class:
        return classification_confidence, classification_confidence_html_class
    return classification_confidence


def get_calibration_reliability_label(is_high_reliability_region, is_ood, sdm_output_for_predicted_class):
    calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_LOW
    if is_high_reliability_region:
        calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_HIGHEST
    elif is_ood:
        calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_OOD
    elif sdm_output_for_predicted_class < constants.CALIBRATION_RELIABILITY_LABEL_LOW__NEAR_CHANCE_THRESHOLD:
        calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_LOW__NEAR_CHANCE
    return calibration_reliability


def format_sdm_estimator_output_for_mcp_tool(prediction_meta_data, gpt5_model_explanation, gemini_model_explanation,
                                             agreement_model_classification: bool):

    predicted_class = prediction_meta_data["prediction"]

    sdm_output_for_predicted_class = \
        prediction_meta_data["sdm_output"].detach().cpu().tolist()[predicted_class]

    verification_classification = predicted_class == 1
    is_high_reliability_region = prediction_meta_data["is_high_reliability_region"]
    # 2026-01-10: Added prediction_meta_data["d"] == 0.0. For these cases, the output is at chance, but the default
    # output to the LM only shows the coarse labels, so this simplifies the interpretation for the tool-calling LM when
    # the full probability vector isn't provided (i.e., without calling the View tool).
    is_ood = prediction_meta_data["is_ood"] or prediction_meta_data["d"] == 0.0
    calibration_reliability = \
        get_calibration_reliability_label(is_high_reliability_region, is_ood,
                                          sdm_output_for_predicted_class=sdm_output_for_predicted_class)

    formatted_output_string = \
        get_formatted_sdm_estimator_output_string(verification_classification,
                                                  calibration_reliability,
                                                  gpt5_model_explanation,
                                                  gemini_model_explanation,
                                                  agreement_model_classification,
                                                  hr_class_conditional_accuracy=
                                                  prediction_meta_data["hr_class_conditional_accuracy"])
    return formatted_output_string


def test(main_device, model, reexpression_input):
    try:
        assert main_device.type == "cpu"

        prediction_meta_data = \
            model(reexpression_input,
                  forward_type=constants.FORWARD_TYPE_SINGLE_PASS_TEST_WITH_EXEMPLAR,
                  return_k_nearest_training_idx_in_prediction_metadata=1)
        # We defer retrieving the training instance from the database, since it is not needed if the
        # visualization is turned off:
        prediction_meta_data["nearest_training_idx"] = prediction_meta_data["top_distance_idx"]
        # add the following model-level values for convenience
        prediction_meta_data["min_rescaled_similarity_to_determine_high_reliability_region"] = \
            model.min_rescaled_similarity_to_determine_high_reliability_region
        prediction_meta_data["hr_output_thresholds"] = model.hr_output_thresholds.detach().cpu().tolist()
        prediction_meta_data["hr_class_conditional_accuracy"] = model.hr_class_conditional_accuracy
        prediction_meta_data["support_index_ntotal"] = model.support_index.ntotal
        return prediction_meta_data
    except:
        return None
