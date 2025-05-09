# Copyright Reexpress AI, Inc. All rights reserved.

# test-time predictions and formatting for MCP server

import torch
import numpy as np

import constants


def get_formatted_sdm_estimator_output_string(verification_classification, confidence_in_verification,
                                              calibration_reliability, log_prob_model_explanation,
                                              reasoning_model_explanation) -> str:
    # If this changes, the docstring in reexpress_mcp_server.reexpress() should also be updated to avoid confusing
    # the downstream LLMs/agents.
    formatted_output_string = f"""
        Successfully Verified: {verification_classification}\n
        Confidence in Successful Verification: {confidence_in_verification}\n
        Calibration Reliability: {calibration_reliability} \n
        Informal Explanation [1]: <model1_explanation> {log_prob_model_explanation} </model1_explanation> \n
        Informal Explanation [2]: <model2_explanation> {reasoning_model_explanation} </model2_explanation>
    """
    return formatted_output_string


def format_sdm_estimator_output_for_mcp_tool(prediction_meta_data, log_prob_model_explanation,
                                             reasoning_model_explanation, ood_limit):
    # SDM() calibrated probabilities are typically not presented as single float values, but existing tool-calling
    # LLMs will not have an understanding of those different quantities, so we compress the additional information
    # (such as the notion of index-conditional estimate validity) into a single value for the purpose of the MCP
    # server. In practice, this means we set a ceiling of 94% (constants.ARBITRARY_NON_INDEX_CONDITIONAL_ESTIMATE_MAX),
    # given that alpha prime is 0.95, on estimates that are not valid index-conditional estimates. Additionally,
    # we refer to OOD estimates as having "Lowest" calibration reliability since existing LLMs may not understand the
    # relative ranking of the phrase "Out-of-distribution (OOD)" vs Highest or Low.
    # (Our own MCP client will directly expose the additional information and nuance to the end-user.)
    # The calibrated probability is relative to class 1 (verified).
    predicted_class = prediction_meta_data["prediction"]

    prediction_conditional_distribution__lower = \
        prediction_meta_data["rescaled_prediction_conditional_distribution__lower"]
    is_valid_index_conditional__lower = prediction_meta_data["is_valid_index_conditional__lower"]

    verification_classification = predicted_class == 1

    prediction_conditional_estimate_of_predicted_class__lower = \
        prediction_conditional_distribution__lower[predicted_class].item()
    if verification_classification:
        prediction_conditional_estimate_of_predicted_class__lower = (
            max(0.5,
                prediction_conditional_estimate_of_predicted_class__lower))
        if not is_valid_index_conditional__lower:
            prediction_conditional_estimate_of_predicted_class__lower = (
                min(constants.ARBITRARY_NON_INDEX_CONDITIONAL_ESTIMATE_MAX,
                    prediction_conditional_estimate_of_predicted_class__lower))
        confidence_in_verification = (
            constants.floatProbToDisplaySignificantDigits(
                floatProb=prediction_conditional_estimate_of_predicted_class__lower))
    else:
        # The lower offset is only applied to the index of the predicted_class, so in this case, the class 1 value
        # is from the original normalized distribution (i.e., there is no need to re-add the lower offset since it
        # has only been applied to class 0).
        prediction_class1__lower__normalized = prediction_conditional_distribution__lower[1].item()
        # The ceiling handles edge cases for OOD/etc. where, e.g., the argmax of the calibrated distribution does
        # not coincide with the argmax of the predicted class, which is rare, but can occur for unreliable/OOD
        # predictions.
        prediction_class1__lower__normalized = (
            min(0.5,
                prediction_class1__lower__normalized))
        confidence_in_verification = (
            constants.floatProbToDisplaySignificantDigits(floatProb=prediction_class1__lower__normalized))

    calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_LOW
    if is_valid_index_conditional__lower:
        calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_HIGHEST
    else:
        hard_qbin_lower = int(prediction_meta_data["soft_qbin__lower"][0].item())
        if hard_qbin_lower <= ood_limit:
            calibration_reliability = constants.CALIBRATION_RELIABILITY_LABEL_OOD
    formatted_output_string = \
        get_formatted_sdm_estimator_output_string(verification_classification, confidence_in_verification,
                                                  calibration_reliability,
                                                  log_prob_model_explanation, reasoning_model_explanation)
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
                  predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin)
        return prediction_meta_data
    except:
        return None
