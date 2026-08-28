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
                                              gpt5_model_explanation,
                                              gemini_model_explanation,
                                              agreement_model_classification: bool | None,
                                              most_conservative_hr_alpha: float,
                                              hr_region_alpha: float) -> str:
    # If this changes, the docstring in reexpress_mcp_server.reexpress() should also be updated to avoid confusing
    # the downstream LLMs/agents. Currently, the docstring is hardcoded for the case where
    # agreement_model_classification is None.
    classification_confidence = \
        get_calibration_confidence_label(hr_region_alpha=hr_region_alpha,
                                         most_conservative_hr_alpha=most_conservative_hr_alpha)
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


def get_calibration_confidence_label(hr_region_alpha: float, most_conservative_hr_alpha: float,
                                     return_html_class=False) -> str:

    if hr_region_alpha == 0.0:
        classification_confidence_html_class = "negative"
        classification_confidence = "Out-of-distribution (unreliable)"
    elif hr_region_alpha == most_conservative_hr_alpha:
        classification_confidence_html_class = "positive"
        classification_confidence = f">= {hr_region_alpha}"
    else:
        classification_confidence_html_class = "caution"
        classification_confidence = f">= {hr_region_alpha}"  # (use with caution)"
    if return_html_class:
        return classification_confidence, classification_confidence_html_class
    return classification_confidence


def format_sdm_estimator_output_for_mcp_tool(prediction_meta_data_dict,
                                             gpt5_model_explanation, gemini_model_explanation,
                                             agreement_model_classification: bool | None = None):
    # Currently only the first model index:
    prediction_meta_data = prediction_meta_data_dict["prediction_meta_data_across_models"][0]

    predicted_class = prediction_meta_data["prediction"]
    verification_classification = predicted_class == 1

    if constants.MCP_SERVER_USE_DKW_LOWER_ESTIMATES:
        hr_region_alpha = prediction_meta_data["hr_region_alpha_lower"]
    else:
        hr_region_alpha = prediction_meta_data["hr_region_alpha"]

    formatted_output_string = \
        get_formatted_sdm_estimator_output_string(verification_classification,
                                                  gpt5_model_explanation,
                                                  gemini_model_explanation,
                                                  agreement_model_classification,
                                                  most_conservative_hr_alpha=
                                                  prediction_meta_data["most_conservative_hr_alpha"],
                                                  hr_region_alpha=hr_region_alpha)
    return formatted_output_string


def random_mode(a):
    # Find the most common value, randomly selecting if there are ties.
    counts = Counter(a)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return random.choice(modes)


def construct_ensemble_prediction(prediction_meta_data_across_models,
                                  most_conservative_hr_alpha_to_consider_in_ensemble=None):
    # This mirrors utils_test_batch_ensemble.py. Currently, len(prediction_meta_data_across_models) == 1

    if len(prediction_meta_data_across_models) == 1:
        return {"ensemble_meta_data": None,
                "prediction_meta_data_across_models": prediction_meta_data_across_models}

    assert most_conservative_hr_alpha_to_consider_in_ensemble is not None and \
           most_conservative_hr_alpha_to_consider_in_ensemble > 0.5
    total_models_in_ensemble = len(prediction_meta_data_across_models)
    predicted_class = random_mode(
        [prediction_meta_data["prediction"] for prediction_meta_data in prediction_meta_data_across_models])

    # Note that we also require all
    # predictions to match in order for the ensemble to be in the HR/HR_lower regions. This is checked in the loop
    # across prediction_meta_data_across_models, below. (.item() is to convert
    # from numpy to int for JSON serialization.)
    is_high_reliability_region_lower = \
        np.sum(
            [prediction_meta_data["hr_region_alpha_lower"] >= most_conservative_hr_alpha_to_consider_in_ensemble
             for prediction_meta_data in prediction_meta_data_across_models]).item() == total_models_in_ensemble
    is_high_reliability_region = \
        np.sum(
            [prediction_meta_data["hr_region_alpha"] >= most_conservative_hr_alpha_to_consider_in_ensemble
             for prediction_meta_data in prediction_meta_data_across_models]).item() == total_models_in_ensemble

    sdm_output = None  # chosen min among predicted_class
    rescaled_similarity = None
    min_sdm_output_index = None
    sdm_output_lower = None  # chosen min among predicted_class
    rescaled_similarity_lower = None
    min_sdm_output_lower_index = None

    q_is_0_or_d_is_0 = False
    hr_region_alpha = None
    hr_region_alpha_index = None
    hr_region_alpha_lower = None
    hr_region_alpha_lower_index = None

    shuffle_index = 0
    for prediction_meta_data in prediction_meta_data_across_models:
        if prediction_meta_data["q"] == 0 or prediction_meta_data["d"] == 0:
            q_is_0_or_d_is_0 = True
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
            if hr_region_alpha is None or \
                    prediction_meta_data["hr_region_alpha"] < hr_region_alpha:
                hr_region_alpha = prediction_meta_data["hr_region_alpha"]
                hr_region_alpha_index = shuffle_index
            if hr_region_alpha_lower is None or \
                    prediction_meta_data["hr_region_alpha_lower"] < hr_region_alpha_lower:
                hr_region_alpha_lower = prediction_meta_data["hr_region_alpha_lower"]
                hr_region_alpha_lower_index = shuffle_index
        else:
            is_high_reliability_region_lower = False
            is_high_reliability_region = False

        shuffle_index += 1

    ensemble_meta_data = {
        # Across models, the modal prediction, with ties randomly broken:
        "ensemble_prediction": predicted_class,
        # All predictions match AND all predictions are in the most conservative HR_lower (where that
        # alpha is specified by the caller to this eval routine and recorded in the field
        # 'most_conservative_hr_alpha_to_consider_in_ensemble'):
        "ensemble_is_high_reliability_region_lower": is_high_reliability_region_lower,
        # Among predictions matching "ensemble_prediction", lowest sdm(z')_lower for the predicted class:
        "ensemble_sdm_output_lower": sdm_output_lower,
        # q'_lower corresponding to the model iteration chosen for "ensemble_sdm_output_lower"
        "ensemble_rescaled_similarity_lower": rescaled_similarity_lower,
        # All predictions match AND all predictions are in the most conservative HR (where that alpha is specified
        # by the caller to this eval routine and recorded in the field
        # 'most_conservative_hr_alpha_to_consider_in_ensemble'):
        "ensemble_is_high_reliability_region": is_high_reliability_region,
        # Among predictions matching "ensemble_prediction", lowest sdm(z') for the predicted class:
        "ensemble_sdm_output": sdm_output,
        # q' corresponding to the model iteration chosen for "ensemble_sdm_output"
        "ensemble_rescaled_similarity": rescaled_similarity,
        # If any of the model predictions have q = 0 or d = 0:
        "ensemble_any_is_q_is_0_or_d_is_0": q_is_0_or_d_is_0,
        # model shuffle index for the min SDM output:
        "min_sdm_output_index": min_sdm_output_index,
        # model shuffle index for the min SDM_lower output:
        "min_sdm_output_lower_index": min_sdm_output_lower_index,
        # Among predictions matching "ensemble_prediction", lowest hr_region_alpha:
        "hr_region_alpha": hr_region_alpha,
        # model shuffle index for the min hr_region_alpha:
        "hr_region_alpha_index": hr_region_alpha_index,
        # Among predictions matching "ensemble_prediction", lowest hr_region_alpha_lower:
        "hr_region_alpha_lower": hr_region_alpha_lower,
        # model shuffle index for the min hr_region_alpha_lower:
        "hr_region_alpha_lower_index": hr_region_alpha_lower_index,
        # This is the alpha value that determines the most conservative HR region under consideration for the
        # selection criteria requiring HR region membership AND prediction matches across all models. Note that
        # hr_region_alpha and hr_region_alpha_lower are less restrictive criteria, as they only consider the modal
        # predicted class.
        "most_conservative_hr_alpha_to_consider_in_ensemble": most_conservative_hr_alpha_to_consider_in_ensemble
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
            hr_region_stats = model.get_most_conservative_high_reliability_region_stats()
            most_conservative_hr_alpha = hr_region_stats["most_conservative_hr_alpha"]
            most_conservative_hr_output_thresholds = hr_region_stats["most_conservative_hr_output_thresholds"]
            most_conservative_hr_min_rescaled_similarity = hr_region_stats[
                "most_conservative_hr_min_rescaled_similarity"]

            prediction_meta_data["most_conservative_hr_alpha"] = \
                most_conservative_hr_alpha
            prediction_meta_data["most_conservative_hr_output_thresholds"] = \
                most_conservative_hr_output_thresholds.detach().cpu().tolist()
            prediction_meta_data["most_conservative_hr_min_rescaled_similarity"] = \
                most_conservative_hr_min_rescaled_similarity

            prediction_meta_data["available_hr_regions"] = \
                [hr_region['alpha'] for hr_region in model.hr_regions]

            prediction_meta_data["support_index_ntotal"] = model.support_index.ntotal

            # 2026-08-22: With the nested regions, we are going to define OOD as any point not assigned a region.
            if constants.MCP_SERVER_USE_DKW_LOWER_ESTIMATES and prediction_meta_data["hr_region_alpha_lower"] == 0.0:
                prediction_meta_data["is_ood"] = True
            if not constants.MCP_SERVER_USE_DKW_LOWER_ESTIMATES and prediction_meta_data["hr_region_alpha"] == 0.0:
                prediction_meta_data["is_ood"] = True

            prediction_meta_data_across_models.append(prediction_meta_data)
        return construct_ensemble_prediction(prediction_meta_data_across_models)
    except:
        return None
