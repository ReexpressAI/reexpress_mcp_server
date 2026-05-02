# Copyright Reexpress AI, Inc. All rights reserved.

# test-time predictions and formatting for MCP server

# import torch
import numpy as np
import random
from collections import Counter

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
                                              agreement_model_classification: bool | None,
                                              hr_class_conditional_accuracy: float) -> str:
    # If this changes, the docstring in reexpress_mcp_server.reexpress() should also be updated to avoid confusing
    # the downstream LLMs/agents. Currently, the docstring is hardcoded for the case where
    # agreement_model_classification is None.
    classification_confidence = \
        get_calibration_confidence_label(calibration_reliability=calibration_reliability,
                                         hr_class_conditional_accuracy=hr_class_conditional_accuracy)
    if agreement_model_classification is not None:
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
    else:
        formatted_output_string = f"""
            <successfully_verified> {verification_classification} </successfully_verified> \n
            <confidence> {classification_confidence} </confidence> \n
            <model1_explanation> {gpt5_model_explanation} </model1_explanation> \n
            <model2_explanation> {gemini_model_explanation} </model2_explanation>
        """
    return formatted_output_string


def get_files_in_consideration_message(attached_files_names_list):
    if len(attached_files_names_list) > 0:
        files_in_consideration_message = f'The verification model had access to: ' \
                                         f'{",".join(attached_files_names_list)}'
    else:
        files_in_consideration_message = f'The verification model did not have access to any external files.'
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


def format_sdm_estimator_output_for_mcp_tool(prediction_meta_data_dict,
                                             gpt5_model_explanation, gemini_model_explanation,
                                             agreement_model_classification: bool | None = None):
    # Currently only the first model index:
    prediction_meta_data = prediction_meta_data_dict["prediction_meta_data_across_models"][0]

    predicted_class = prediction_meta_data["prediction"]
    verification_classification = predicted_class == 1

    if constants.MCP_SERVER_USE_DKW_LOWER_ESTIMATES:
        sdm_output_for_predicted_class = \
            prediction_meta_data["sdm_output_d_lower"].detach().cpu().tolist()[predicted_class]
        is_high_reliability_region = prediction_meta_data["is_high_reliability_region_lower"]
    else:
        sdm_output_for_predicted_class = \
            prediction_meta_data["sdm_output"].detach().cpu().tolist()[predicted_class]
        is_high_reliability_region = prediction_meta_data["is_high_reliability_region"]

    # OOD also takes into account d == 0. (See note in mcp_utils_test.test().)
    is_ood = prediction_meta_data["is_ood"]
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


def random_mode(a):
    # Find the most common value, randomly selecting if there are ties.
    counts = Counter(a)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return random.choice(modes)


def construct_ensemble_prediction(prediction_meta_data_across_models):
    # This mirrors utils_test_batch_ensemble.py

    if len(prediction_meta_data_across_models) == 1:
        return {"ensemble_meta_data": None,
                "prediction_meta_data_across_models": prediction_meta_data_across_models}

    total_models_in_ensemble = len(prediction_meta_data_across_models)
    predicted_class = random_mode(
        [prediction_meta_data["prediction"] for prediction_meta_data in prediction_meta_data_across_models])

    # Note that we also require all
    # predictions to match in order for the ensemble to be in the HR/HR_lower regions. This is checked in the loop
    # across prediction_meta_data_across_models, below. (.item() is to convert
    # from numpy to int for JSON serialization.)
    is_high_reliability_region_lower = \
        np.sum(
            [prediction_meta_data["is_high_reliability_region_lower"]
             for prediction_meta_data in prediction_meta_data_across_models]).item() == total_models_in_ensemble
    is_high_reliability_region = \
        np.sum(
            [prediction_meta_data["is_high_reliability_region"]
             for prediction_meta_data in prediction_meta_data_across_models]).item() == total_models_in_ensemble

    is_ood = False
    sdm_output = None  # chosen min among predicted_class
    rescaled_similarity = None
    min_sdm_output_index = None
    sdm_output_lower = None  # chosen min among predicted_class
    rescaled_similarity_lower = None
    min_sdm_output_lower_index = None
    shuffle_index = 0
    for prediction_meta_data in prediction_meta_data_across_models:
        # OOD also takes into account d == 0. (See note in mcp_utils_test.test().)
        if prediction_meta_data["is_ood"]:
            # OOD if at least one OOD
            is_ood = True
        if prediction_meta_data["prediction"] == predicted_class:
            if sdm_output is None or \
                    prediction_meta_data["sdm_output"][predicted_class] < sdm_output[predicted_class]:
                sdm_output = prediction_meta_data["sdm_output"]
                rescaled_similarity = prediction_meta_data["rescaled_similarity"]
                min_sdm_output_index = shuffle_index
            if sdm_output_lower is None or \
                    prediction_meta_data["sdm_output_d_lower"][predicted_class] < sdm_output_lower[predicted_class]:
                sdm_output_lower = prediction_meta_data["sdm_output_d_lower"]
                rescaled_similarity_lower = prediction_meta_data["rescaled_similarity_lower"]
                min_sdm_output_lower_index = shuffle_index
        else:
            is_high_reliability_region_lower = False
            is_high_reliability_region = False

        shuffle_index += 1

    ensemble_meta_data = {
        # Across models, the modal prediction, with ties randomly broken:
        "ensemble_prediction": predicted_class,
        # All predictions match AND all predictions are in HR_lower:
        "ensemble_is_high_reliability_region_lower": is_high_reliability_region_lower,
        # Among predictions matching "ensemble_prediction", lowest sdm(z')_lower for the predicted class:
        "ensemble_sdm_output_lower": sdm_output_lower,
        # q'_lower corresponding to the model iteration chosen for "ensemble_sdm_output_lower"
        "ensemble_rescaled_similarity_lower": rescaled_similarity_lower,
        # All predictions match AND all predictions are in HR:
        "ensemble_is_high_reliability_region": is_high_reliability_region,
        # Among predictions matching "ensemble_prediction", lowest sdm(z') for the predicted class:
        "ensemble_sdm_output": sdm_output,
        # q' corresponding to the model iteration chosen for "ensemble_sdm_output"
        "ensemble_rescaled_similarity": rescaled_similarity,
        # If any of the model predictions are OOD:
        "ensemble_any_is_ood": is_ood,
        # model shuffle index for the min SDM output:
        "min_sdm_output_index": min_sdm_output_index,
        # model shuffle index for the min SDM_lower output:
        "min_sdm_output_lower_index": min_sdm_output_lower_index
    }
    json_obj = {"ensemble_meta_data": ensemble_meta_data,
                "prediction_meta_data_across_models": prediction_meta_data_across_models}
    return json_obj


def test(main_device, model_list, reexpression_input):
    try:
        assert main_device.type == "cpu"
        prediction_meta_data_across_models = []
        for model in model_list:
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

            # 2026-04-30: Override is_ood to also take into account a distance quantile of 0.
            # For these cases, the output is at chance, but the default
            # output to the LM only shows the coarse labels, so this simplifies the interpretation for
            # the tool-calling LM when
            # the full probability vector isn't provided (i.e., without calling the View tool).
            if constants.MCP_SERVER_USE_DKW_LOWER_ESTIMATES and prediction_meta_data["d_lower"] == 0.0:
                prediction_meta_data["is_ood"] = True
            if not constants.MCP_SERVER_USE_DKW_LOWER_ESTIMATES and prediction_meta_data["d"] == 0.0:
                prediction_meta_data["is_ood"] = True

            prediction_meta_data_across_models.append(prediction_meta_data)
        return construct_ensemble_prediction(prediction_meta_data_across_models)
    except:
        return None
