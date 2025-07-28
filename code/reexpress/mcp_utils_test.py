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
                                              calibration_reliability, log_prob_model_explanation,
                                              reasoning_model_explanation,
                                              gemini_model_explanation,
                                              agreement_model_classification: bool,
                                              non_odd_class_conditional_accuracy: float) -> str:
    # If this changes, the docstring in reexpress_mcp_server.reexpress() should also be updated to avoid confusing
    # the downstream LLMs/agents.
    classification_confidence = \
        get_calibration_confidence_label(calibration_reliability=calibration_reliability,
                                         non_odd_class_conditional_accuracy=non_odd_class_conditional_accuracy)
    if agreement_model_classification:
        agreement_model_classification_string = "Yes"
    else:
        agreement_model_classification_string = "No"
    formatted_output_string = f"""
        <successfully_verified> {verification_classification} </successfully_verified> \n
        <confidence> {classification_confidence} </confidence> \n
        <model1_explanation> {log_prob_model_explanation} </model1_explanation> \n
        <model2_explanation> {reasoning_model_explanation} </model2_explanation> \n
        <model3_explanation> {gemini_model_explanation} </model3_explanation> \n
        <model4_agreement> {constants.AGREEMENT_MODEL_USER_FACING_PROMPT} {agreement_model_classification_string} </model4_agreement>
    """
    return formatted_output_string


def get_files_in_consideration_message(attached_files_names_list):
    if len(attached_files_names_list) > 0:
        files_in_consideration_message = f'The verification model had access to: ' \
                                         f'{",".join(attached_files_names_list)}\n\n'
    else:
        files_in_consideration_message = f'The verification model did not have access to any external files.\n\n'
    return files_in_consideration_message


def get_calibration_confidence_label(calibration_reliability: str, non_odd_class_conditional_accuracy: float,
                                     return_html_class=False) -> str:

    if calibration_reliability == constants.CALIBRATION_RELIABILITY_LABEL_OOD:
        classification_confidence_html_class = "negative"
        classification_confidence = "Out-of-distribution (unreliable)"
    elif calibration_reliability == constants.CALIBRATION_RELIABILITY_LABEL_HIGHEST:
        classification_confidence_html_class = "positive"
        classification_confidence = f">= {_format_probability_as_string_percentage(valid_probability_float=non_odd_class_conditional_accuracy)}"
    else:
        classification_confidence_html_class = "caution"
        classification_confidence = f"< {_format_probability_as_string_percentage(valid_probability_float=non_odd_class_conditional_accuracy)} (use with caution)"
    if return_html_class:
        return classification_confidence, classification_confidence_html_class
    return classification_confidence


def get_calibration_reliability_label(is_valid_index_conditional__lower, is_ood_lower):
    calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_LOW
    if is_valid_index_conditional__lower:
        calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_HIGHEST
    elif is_ood_lower:
        calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_OOD
    return calibration_reliability


def format_sdm_estimator_output_for_mcp_tool(prediction_meta_data, log_prob_model_explanation,
                                             reasoning_model_explanation, gemini_model_explanation,
                                             agreement_model_classification: bool):
    # Starting in v1.1.0, we've streamlined the primary output to the information needed in (and at a
    # default granularity suitable for) typical applications.
    predicted_class = prediction_meta_data["prediction"]

    # prediction_conditional_distribution__lower = \
    #     prediction_meta_data["rescaled_prediction_conditional_distribution__lower"]

    verification_classification = predicted_class == 1
    is_valid_index_conditional__lower = prediction_meta_data["is_valid_index_conditional__lower"]
    is_ood_lower = prediction_meta_data["is_ood_lower"]
    calibration_reliability = \
        get_calibration_reliability_label(is_valid_index_conditional__lower, is_ood_lower)

    formatted_output_string = \
        get_formatted_sdm_estimator_output_string(verification_classification,
                                                  calibration_reliability,
                                                  log_prob_model_explanation,
                                                  reasoning_model_explanation,
                                                  gemini_model_explanation,
                                                  agreement_model_classification,
                                                  non_odd_class_conditional_accuracy=
                                                  prediction_meta_data["non_odd_class_conditional_accuracy"])
    return formatted_output_string


def test(main_device, model, global_uncertainty_statistics, reexpression_input):
    try:
        assert main_device.type == "cpu"
        min_valid_qbin_for_class_conditional_accuracy_with_bounded_error = \
            global_uncertainty_statistics.get_min_valid_qbin_with_bounded_error(
                model.min_valid_qbin_for_class_conditional_accuracy)

        predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin = \
            global_uncertainty_statistics.get_summarized_output_magnitude_structure_with_bounded_error_lower_offset_by_bin()

        prediction_meta_data = \
            model(reexpression_input,
                  forward_type=constants.FORWARD_TYPE_SINGLE_PASS_TEST_WITH_EXEMPLAR,
                  min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=
                  min_valid_qbin_for_class_conditional_accuracy_with_bounded_error,
                  predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=
                  predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin,
                  return_k_nearest_training_idx_in_prediction_metadata=1)
        nearest_training_idx = int(prediction_meta_data["top_k_distances_idx"][0])
        # We defer retrieving the training instance from the database, since it is not needed if the
        # visualization is turned off:
        prediction_meta_data["nearest_training_idx"] = nearest_training_idx
        # add the following model-level values for convenience
        prediction_meta_data["min_valid_qbin_for_class_conditional_accuracy_with_bounded_error"] = \
            min_valid_qbin_for_class_conditional_accuracy_with_bounded_error
        prediction_meta_data["non_odd_thresholds"] = model.non_odd_thresholds.tolist()
        prediction_meta_data["non_odd_class_conditional_accuracy"] = model.non_odd_class_conditional_accuracy
        prediction_meta_data["support_index_ntotal"] = model.support_index.ntotal
        return prediction_meta_data
    except:
        return None
