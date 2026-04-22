# Copyright Reexpress AI, Inc. All rights reserved.
import copy
import torch
import numpy as np
import random
from collections import Counter
from collections import namedtuple

import constants
import utils_model
import sdm_model
import utils_preprocess
import data_validator
import utils_eval_batch

from utils_latex import print_latex_row, init_latex_rows_dict, get_acc_prop_tuple, MARGINAL_ACC_KEY, \
    MARGINAL_ADMITTED_KEY, CLASS_CONDITIONAL_ACC_KEY, CLASS_CONDITIONAL_ADMITTED_KEY, PREDICTION_CONDITIONAL_ACC_KEY, \
    PREDICTION_CONDITIONAL_ADMITTED_KEY

from utils_test_batch import print_summary, get_bin


def random_mode(a):
    # Find the most common value, randomly selecting if there are ties.
    counts = Counter(a)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return random.choice(modes)


def test(options, id2ensemble_stats=None, numberOfClasses=2, maxQAvailableFromIndexer=2048, total_models_in_ensemble=0):
    random.seed(options.seed_value)
    assert id2ensemble_stats is not None
    print(f"###############Beginning of Ensemble evaluation ({total_models_in_ensemble} models)###############")

    alpha_prime = options.alpha
    print(f"alpha={alpha_prime}")
    test_set_size = len(id2ensemble_stats)

    latex_rows_dict_no_reject = init_latex_rows_dict(numberOfClasses=numberOfClasses)
    latex_rows_dict_sdm = init_latex_rows_dict(numberOfClasses=numberOfClasses)
    latex_rows_dict_sdm_hr = init_latex_rows_dict(numberOfClasses=numberOfClasses)
    latex_rows_dict_sdm_hr_lower = init_latex_rows_dict(numberOfClasses=numberOfClasses)

    q_val_rescaled_by_sdm_by_classConditionalAccuracy = []
    q_val_rescaled_by_sdm_by_predictionConditionalAccuracy = []

    for q in range(maxQAvailableFromIndexer+1):
        q_val_rescaled_by_sdm_by_classConditionalAccuracy.append({})
        q_val_rescaled_by_sdm_by_predictionConditionalAccuracy.append({})
        for trueLabel in range(numberOfClasses):
            q_val_rescaled_by_sdm_by_classConditionalAccuracy[q][trueLabel] = []
            q_val_rescaled_by_sdm_by_predictionConditionalAccuracy[q][trueLabel] = []

    marginal_accuracy = []
    marginal_accuracy_filtered__sdm_by_hr_region = []
    marginal_accuracy_filtered__sdm_by_hr_region_lower = []
    marginal_accuracy_filtered__sdm_by_alpha_prime = []

    class_conditional_accuracy = {}
    class_conditional_accuracy_filtered__sdm_by_hr_region = {}
    class_conditional_accuracy_filtered__sdm_by_hr_region_lower = {}
    class_conditional_accuracy_filtered__sdm_by_alpha_prime = {}

    class_conditional_accuracy__is_ood_sdm = {}
    class_conditional_accuracy__NOT_is_ood__AND__NOT_is_hr_region = {}

    prediction_conditional_accuracy = {}
    prediction_conditional_accuracy_filtered__sdm_by_hr_region = {}
    prediction_conditional_accuracy_filtered__sdm_by_hr_region_lower = {}
    prediction_conditional_accuracy_filtered__sdm_by_alpha_prime = {}
    for label in range(numberOfClasses):
        class_conditional_accuracy[label] = []
        class_conditional_accuracy_filtered__sdm_by_hr_region[label] = []
        class_conditional_accuracy_filtered__sdm_by_hr_region_lower[label] = []
        class_conditional_accuracy_filtered__sdm_by_alpha_prime[label] = []
        class_conditional_accuracy__is_ood_sdm[label] = []
        class_conditional_accuracy__NOT_is_ood__AND__NOT_is_hr_region[label] = []

        prediction_conditional_accuracy[label] = []
        prediction_conditional_accuracy_filtered__sdm_by_hr_region[label] = []
        prediction_conditional_accuracy_filtered__sdm_by_hr_region_lower[label] = []
        prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[label] = []

    possible_label_error_json_lines = []  # HR (i.e., centroid)
    predictions_in_high_reliability_region_json_lines = []  # HR
    possible_label_error_json_lines_lower = []  # HR_lower
    predictions_in_high_reliability_region_json_lines_lower = []  # HR_lower
    all_predictions_json_lines = []
    number_of_divisions = 20
    predicted_f_binned = [x for x in range(number_of_divisions)]
    true_frequency_binned = [[] for x in range(number_of_divisions)]

    true_frequency_binned_prediction_conditional = {}
    true_frequency_binned_prediction_conditional__average_sample_sizes = {}
    true_frequency_binned_class_conditional = {}
    for label in range(numberOfClasses):
        true_frequency_binned_prediction_conditional[label] = [[] for x in range(number_of_divisions)]
        true_frequency_binned_prediction_conditional__average_sample_sizes[label] = \
            [[] for x in range(number_of_divisions)]
        true_frequency_binned_class_conditional[label] = [[] for x in range(number_of_divisions)]
    instance_i = -1

    for document_id in id2ensemble_stats:
        prediction_meta_data_across_models = id2ensemble_stats[document_id]
        assert len(prediction_meta_data_across_models) == total_models_in_ensemble
        instance_i += 1
        true_test_label = prediction_meta_data_across_models[0][constants.REEXPRESS_LABEL_KEY]
        assert data_validator.isValidLabel(label=true_test_label, numberOfClasses=numberOfClasses)
        assert data_validator.isKnownValidLabel(label=true_test_label, numberOfClasses=numberOfClasses)
        assert total_models_in_ensemble == len(prediction_meta_data_across_models)
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
        cumulative_effective_sample_sizes = None  # placeholder
        floor_rescaled_similarity = None  # placeholder
        shuffle_index = 0
        for prediction_meta_data in prediction_meta_data_across_models:
            assert true_test_label == prediction_meta_data[constants.REEXPRESS_LABEL_KEY]
            assert document_id == prediction_meta_data[constants.REEXPRESS_ID_KEY]
            if prediction_meta_data["is_ood"]:
                # OOD if at least one OOD
                is_ood = True
            if prediction_meta_data["prediction"] == predicted_class:
                if sdm_output is None or \
                        prediction_meta_data["sdm_output"][predicted_class] < sdm_output[predicted_class]:
                    sdm_output = prediction_meta_data["sdm_output"]
                    rescaled_similarity = prediction_meta_data["rescaled_similarity"]
                    min_sdm_output_index = shuffle_index
                    # The following 2 vars are just to push through the exiting eval code and
                    # are effectively placeholders:
                    cumulative_effective_sample_sizes = prediction_meta_data["cumulative_effective_sample_sizes"]
                    floor_rescaled_similarity = prediction_meta_data["floor_rescaled_similarity"]
                if sdm_output_lower is None or \
                        prediction_meta_data["sdm_output_d_lower"][predicted_class] < sdm_output_lower[predicted_class]:
                    sdm_output_lower = prediction_meta_data["sdm_output_d_lower"]
                    rescaled_similarity_lower = prediction_meta_data["rescaled_similarity_lower"]
                    min_sdm_output_lower_index = shuffle_index
            else:
                is_high_reliability_region_lower = False
                is_high_reliability_region = False

            shuffle_index += 1

        ensemble_json_obj = {
            constants.REEXPRESS_ID_KEY: document_id,
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
        json_obj = [ensemble_json_obj] + id2ensemble_stats[document_id]
        all_predictions_json_lines.append(json_obj)

        prediction_conditional_distribution__sdm = \
            sdm_output

        prediction_conditional_estimate_of_predicted_class__sdm = \
            prediction_conditional_distribution__sdm[predicted_class]

        prediction_conditional_estimate_of_predicted_class__sdm_lower = \
            sdm_output_lower[predicted_class]  # using DKW

        q_val_rescaled_by_sdm_by_classConditionalAccuracy[floor_rescaled_similarity][true_test_label].append(
            predicted_class == true_test_label)
        q_val_rescaled_by_sdm_by_predictionConditionalAccuracy[floor_rescaled_similarity][predicted_class].append(
            predicted_class == true_test_label)

        marginal_accuracy.append(predicted_class == true_test_label)
        class_conditional_accuracy[true_test_label].append(predicted_class == true_test_label)
        prediction_conditional_accuracy[predicted_class].append(predicted_class == true_test_label)

        if is_ood:
            class_conditional_accuracy__is_ood_sdm[true_test_label].append(predicted_class == true_test_label)
        if not is_ood and not is_high_reliability_region:
            class_conditional_accuracy__NOT_is_ood__AND__NOT_is_hr_region[true_test_label].append(
                predicted_class == true_test_label)

        if is_high_reliability_region:  # HR region
            class_conditional_accuracy_filtered__sdm_by_hr_region[true_test_label].append(
                predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__sdm_by_hr_region[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__sdm_by_hr_region.append(predicted_class == true_test_label)
            # first two elements are for sorting before saving
            predictions_in_high_reliability_region_json_lines.append(
                (prediction_conditional_estimate_of_predicted_class__sdm,
                 rescaled_similarity,
                 json_obj))
            if predicted_class != true_test_label:
                possible_label_error_json_lines.append(
                    (prediction_conditional_estimate_of_predicted_class__sdm,
                     rescaled_similarity,
                     json_obj))
        if is_high_reliability_region_lower:  # HR_lower region
            class_conditional_accuracy_filtered__sdm_by_hr_region_lower[true_test_label].append(
                predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__sdm_by_hr_region_lower[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__sdm_by_hr_region_lower.append(predicted_class == true_test_label)
            # first two elements are for sorting before saving
            predictions_in_high_reliability_region_json_lines_lower.append(
                (prediction_conditional_estimate_of_predicted_class__sdm_lower,
                 rescaled_similarity_lower,
                 json_obj))
            if predicted_class != true_test_label:
                possible_label_error_json_lines_lower.append(
                    (prediction_conditional_estimate_of_predicted_class__sdm_lower,
                     rescaled_similarity_lower,
                     json_obj))

        if prediction_conditional_estimate_of_predicted_class__sdm >= alpha_prime:
            class_conditional_accuracy_filtered__sdm_by_alpha_prime[true_test_label].append(
                predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__sdm_by_alpha_prime.append(predicted_class == true_test_label)

        prediction_conditional_estimate_binned = \
            get_bin(prediction_conditional_estimate_of_predicted_class__sdm, divisions=number_of_divisions)
        true_frequency_binned[prediction_conditional_estimate_binned].append(predicted_class == true_test_label)
        true_frequency_binned_prediction_conditional[predicted_class][prediction_conditional_estimate_binned].append(
            predicted_class == true_test_label)
        true_frequency_binned_prediction_conditional__average_sample_sizes[predicted_class][
            prediction_conditional_estimate_binned].extend(
            cumulative_effective_sample_sizes)
        true_frequency_binned_class_conditional[true_test_label][prediction_conditional_estimate_binned].append(
            predicted_class == true_test_label)

    print(f"######## Conditional estimates ########")
    print(f"\tLegend: 'HR': High Reliability region")
    for label in range(numberOfClasses):
        print(f"Label {label} ---")
        print_summary(f"Class-conditional accuracy: Label {label}",
                      class_conditional_accuracy[label], total=test_set_size)
        latex_rows_dict_no_reject[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_no_reject[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy[label], total=test_set_size)

        print_summary(f"\t**Class-conditional HR LOWER accuracy: \t\tLabel {label}",
                      class_conditional_accuracy_filtered__sdm_by_hr_region_lower[label], total=test_set_size)
        latex_rows_dict_sdm_hr_lower[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_sdm_hr_lower[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy_filtered__sdm_by_hr_region_lower[label], total=test_set_size)

        print_summary(f"\t**Class-conditional HR accuracy: \t\tLabel {label}",
                      class_conditional_accuracy_filtered__sdm_by_hr_region[label], total=test_set_size)
        latex_rows_dict_sdm_hr[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_sdm_hr[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy_filtered__sdm_by_hr_region[label], total=test_set_size)

        print_summary(f"\t>>Class-conditional SDM_predicted >= {alpha_prime} accuracy: \t\tLabel {label}",
                      class_conditional_accuracy_filtered__sdm_by_alpha_prime[label], total=test_set_size)
        latex_rows_dict_sdm[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_sdm[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy_filtered__sdm_by_alpha_prime[label], total=test_set_size)

        print_summary(f"Prediction-conditional accuracy: Label {label}",
                      prediction_conditional_accuracy[label], total=test_set_size)
        latex_rows_dict_no_reject[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_no_reject[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy[label], total=test_set_size)

        print_summary(f"\t**Prediction-conditional HR LOWER accuracy: "
                      f"\t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__sdm_by_hr_region_lower[label], total=test_set_size)
        latex_rows_dict_sdm_hr_lower[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_sdm_hr_lower[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy_filtered__sdm_by_hr_region_lower[label],
                               total=test_set_size)

        print_summary(f"\t**Prediction-conditional HR accuracy: "
                      f"\t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__sdm_by_hr_region[label], total=test_set_size)
        latex_rows_dict_sdm_hr[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_sdm_hr[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy_filtered__sdm_by_hr_region[label], total=test_set_size)

        print_summary(f"\t>>Prediction-conditional SDM_predicted >= {alpha_prime} accuracy: "
                      f"\t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[label], total=test_set_size)
        latex_rows_dict_sdm[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_sdm[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[label], total=test_set_size)

    print(f"######## Class-Conditional estimates for non-HR instances that are NOT OOD ########")
    for label in range(numberOfClasses):
        print(f"Label {label} ---")
        print_summary(f"Class-conditional accuracy (not HR AND not OOD): Label {label}",
                      class_conditional_accuracy__NOT_is_ood__AND__NOT_is_hr_region[label],
                      total=test_set_size)
    print(f"######## Class-Conditional estimates for OOD ########")
    for label in range(numberOfClasses):
        print(f"Label {label} ---")
        print_summary(f"Class-conditional accuracy (OOD): Label {label}",
                      class_conditional_accuracy__is_ood_sdm[label],
                      total=test_set_size)

    print(f"######## Stratified by probability, sdm(z') ########")
    for bin in predicted_f_binned:
        print_summary(f"{bin/number_of_divisions}-{(min(number_of_divisions, bin+1))/number_of_divisions}: "
                      f"PREDICTION CONDITIONAL: Marginal",
                      true_frequency_binned[bin])
        for label in range(numberOfClasses):
            print(
                f"\tLabel {label} PREDICTION CONDITIONAL: "
                f"{np.mean(true_frequency_binned_prediction_conditional[label][bin])}, "
                f"out of {len(true_frequency_binned_prediction_conditional[label][bin])} || "
                f"mean sample size: "
                f"{np.mean(true_frequency_binned_prediction_conditional__average_sample_sizes[label][bin])} || "
                f"median sample size: "
                f"{np.median(true_frequency_binned_prediction_conditional__average_sample_sizes[label][bin])}")
            print(
                f"\tLabel {label} -class- -conditional-: "
                f"{np.mean(true_frequency_binned_class_conditional[label][bin])}, "
                f"out of {len(true_frequency_binned_class_conditional[label][bin])}")

    print(f"######## Stratified by floor of the rescaled Similarity (q') ########")
    for q in range(maxQAvailableFromIndexer+1):
        for label in range(numberOfClasses):
            if len(q_val_rescaled_by_sdm_by_classConditionalAccuracy[q][label]) > 0:
                print(f"floor(q'): {q}, label: {label}: class conditional accuracy: \t"
                      f"{np.mean(q_val_rescaled_by_sdm_by_classConditionalAccuracy[q][label])} "
                      f"out of {len(q_val_rescaled_by_sdm_by_classConditionalAccuracy[q][label])}")

            if len(q_val_rescaled_by_sdm_by_predictionConditionalAccuracy[q][label]) > 0:
                print(f"floor(q'): {q}, label: {label}: prediction conditional accuracy: \t"
                      f"{np.mean(q_val_rescaled_by_sdm_by_predictionConditionalAccuracy[q][label])} "
                      f"out of {len(q_val_rescaled_by_sdm_by_predictionConditionalAccuracy[q][label])}")
    print(f"######## Marginal estimates ########")
    print(f"Marginal accuracy: {np.mean(marginal_accuracy)} out of {len(marginal_accuracy)}")
    latex_rows_dict_no_reject[MARGINAL_ACC_KEY], \
        latex_rows_dict_no_reject[MARGINAL_ADMITTED_KEY] = \
        get_acc_prop_tuple(marginal_accuracy, total=len(marginal_accuracy))

    if len(marginal_accuracy) > 0:  # it could be 0 if the eval file only includes OOD or unlabeled
        print(
            f"Filtered HR_lower marginal (constrained to the high reliability LOWER region): "
            f"{np.mean(marginal_accuracy_filtered__sdm_by_hr_region_lower)} out of "
            f"{len(marginal_accuracy_filtered__sdm_by_hr_region_lower)} "
            f"({len(marginal_accuracy_filtered__sdm_by_hr_region_lower)/len(marginal_accuracy)})")
        latex_rows_dict_sdm_hr_lower[MARGINAL_ACC_KEY], \
            latex_rows_dict_sdm_hr_lower[MARGINAL_ADMITTED_KEY] = \
            get_acc_prop_tuple(marginal_accuracy_filtered__sdm_by_hr_region_lower, total=len(marginal_accuracy))

        print(
            f"Filtered HR marginal (constrained to the high reliability region): "
            f"{np.mean(marginal_accuracy_filtered__sdm_by_hr_region)} out of "
            f"{len(marginal_accuracy_filtered__sdm_by_hr_region)} "
            f"({len(marginal_accuracy_filtered__sdm_by_hr_region)/len(marginal_accuracy)})")
        latex_rows_dict_sdm_hr[MARGINAL_ACC_KEY], \
            latex_rows_dict_sdm_hr[MARGINAL_ADMITTED_KEY] = \
            get_acc_prop_tuple(marginal_accuracy_filtered__sdm_by_hr_region, total=len(marginal_accuracy))

        print(
            f"Filtered marginal (constrained to SDM_predicted >= {alpha_prime}): "
            f"{np.mean(marginal_accuracy_filtered__sdm_by_alpha_prime)} out of "
            f"{len(marginal_accuracy_filtered__sdm_by_alpha_prime)} "
            f"({len(marginal_accuracy_filtered__sdm_by_alpha_prime)/len(marginal_accuracy)})")
        latex_rows_dict_sdm[MARGINAL_ACC_KEY], \
            latex_rows_dict_sdm[MARGINAL_ADMITTED_KEY] = \
            get_acc_prop_tuple(marginal_accuracy_filtered__sdm_by_alpha_prime, total=len(marginal_accuracy))

    print(f"######## ########")
    # Save error and admitted predictions for HR_lower
    possible_label_error_json_lines_lower = [y[2] for y in sorted(possible_label_error_json_lines_lower,
                                                                  key=lambda x: (x[0], x[1]),
                                                                  reverse=True)]
    if options.eval_ensemble_label_error_hr_lower_file != "" and len(possible_label_error_json_lines_lower) > 0:
        utils_model.save_json_lines(options.eval_ensemble_label_error_hr_lower_file,
                                    possible_label_error_json_lines_lower)
        print(f">ENSEMBLE: {len(possible_label_error_json_lines_lower)} candidate label errors "
              f"(in HR_lower but y != prediction) saved to {options.eval_ensemble_label_error_hr_lower_file}")

    predictions_in_high_reliability_region_json_lines_lower = \
        [y[2] for y in sorted(predictions_in_high_reliability_region_json_lines_lower,
                              key=lambda x: (x[0], x[1]),
                              reverse=True)]
    if options.eval_ensemble_predictions_in_high_reliability_region_lower_file != "" and \
            len(predictions_in_high_reliability_region_json_lines_lower) > 0:
        utils_model.save_json_lines(options.eval_ensemble_predictions_in_high_reliability_region_lower_file,
                                    predictions_in_high_reliability_region_json_lines_lower)
        print(f">ENSEMBLE: {len(predictions_in_high_reliability_region_json_lines_lower)} "
              f"high reliability predictions (in HR_lower) saved to "
              f"{options.eval_ensemble_predictions_in_high_reliability_region_lower_file}")

    # Additionally save error and admitted predictions for HR (centroid)
    possible_label_error_json_lines = [y[2] for y in sorted(possible_label_error_json_lines,
                                                            key=lambda x: (x[0], x[1]),
                                                            reverse=True)]
    if options.eval_ensemble_label_error_file != "" and len(possible_label_error_json_lines) > 0:
        utils_model.save_json_lines(options.eval_ensemble_label_error_file, possible_label_error_json_lines)
        print(f">ENSEMBLE: {len(possible_label_error_json_lines)} candidate label errors "
              f"(in HR but y != prediction) saved to {options.eval_ensemble_label_error_file}")

    predictions_in_high_reliability_region_json_lines = \
        [y[2] for y in sorted(predictions_in_high_reliability_region_json_lines,
                              key=lambda x: (x[0], x[1]),
                              reverse=True)]
    if options.eval_ensemble_predictions_in_high_reliability_region_file != "" and \
            len(predictions_in_high_reliability_region_json_lines) > 0:
        utils_model.save_json_lines(options.eval_ensemble_predictions_in_high_reliability_region_file,
                                    predictions_in_high_reliability_region_json_lines)
        print(f">ENSEMBLE: {len(predictions_in_high_reliability_region_json_lines)} "
              f"high reliability predictions (in HR) saved to "
              f"{options.eval_ensemble_predictions_in_high_reliability_region_file}")

    # All predictions file:
    if options.eval_ensemble_prediction_output_file != "" and len(all_predictions_json_lines) > 0:
        utils_model.save_json_lines(options.eval_ensemble_prediction_output_file, all_predictions_json_lines)
        print(f">ENSEMBLE: The prediction for each document (total: {len(all_predictions_json_lines)}) "
              f"has been saved to {options.eval_ensemble_prediction_output_file}")

    assert test_set_size == instance_i + 1, "ERROR: The index is mismatched."

    PlaceholderModel = namedtuple('PlaceholderModel', ['numberOfClasses'])
    placeholder_model = PlaceholderModel(numberOfClasses=numberOfClasses)
    print_latex_row(options, placeholder_model, alpha_prime,
                    latex_rows_dict_no_reject, None,
                    None, latex_rows_dict_sdm, latex_rows_dict_sdm_hr,
                    latex_rows_dict_sdm_hr_lower)
