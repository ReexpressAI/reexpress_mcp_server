# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import torch.nn as nn

import numpy as np
import argparse
import copy
from pathlib import Path
import math

from collections import defaultdict

import codecs
import time

import json
import copy
import os

import utils_train_main
import utils_classification
import uncertainty_statistics
import uuid
import constants
import utils_model
import sdm_model
import utils_gen
import utils_train_main_gen_ai_router
import utils_preprocess

from mlx_lm import load

import data_validator


def get_bin(x_val_in_01, divisions=10):
    return int(np.floor(min(0.99, x_val_in_01) * divisions) % divisions)


def print_summary(header_label, list_to_process, total=None):
    if total is not None and total > 0:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)} "
            f"\t({len(list_to_process)/total})% of {total}")
    else:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)}")


def test(options, main_device):
    model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"), load_for_inference=True)
    global_uncertainty_statistics = utils_model.load_global_uncertainty_statistics_from_disk(options.model_dir)

    if model.calibration_training_stage == sdm_model.modelCalibrationTrainingStages.init:
        print(f"The model has not been trained. Exiting.")
        exit()
    if model.calibration_training_stage != sdm_model.modelCalibrationTrainingStages.complete:
        print(f"The model rescaler has not been trained. Train from scratch or run --options.train_rescaler. Exiting.")
        exit()

    if model.alpha != options.alpha:
        print(f"The alpha value used for calibration was {model.alpha}, but "
              f"{options.alpha} was requested. Run --options.train_rescaler to recalibrate based on the new value. "
              f"Optionally include --only_update_rescaler_alpha to only update alpha and not update the rescaler "
              f"weights."
              f"Exiting.")
        exit()

    print(f"Min valid qbin for best model: {model.min_valid_qbin_for_class_conditional_accuracy}")
    print(f"Min valid qbin Median absolute deviation around the median: "
          f"{global_uncertainty_statistics._get_min_valid_qbin_mad()}")
    print(f"all min qbins: {global_uncertainty_statistics.min_valid_qbin_across_iterations}")

    min_valid_qbin_for_class_conditional_accuracy_with_bounded_error = \
        global_uncertainty_statistics.get_min_valid_qbin_with_bounded_error(
        model.min_valid_qbin_for_class_conditional_accuracy)
    print(f"min_valid_qbin_for_class_conditional_accuracy_with_bounded_error: "
          f"{min_valid_qbin_for_class_conditional_accuracy_with_bounded_error}")

    predicted_class_to_bin_to_output_magnitude_mad = \
        global_uncertainty_statistics._get_summarized_output_magnitude_structure()
    predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin = \
        global_uncertainty_statistics.get_summarized_output_magnitude_structure_with_bounded_error_lower_offset_by_bin()
    print(f"Output Magnitude: Median absolute deviation around the median:")
    for label in range(model.numberOfClasses):
        for hard_bin in range(constants.default_max_hard_bin):
            if label in predicted_class_to_bin_to_output_magnitude_mad and \
                    hard_bin in predicted_class_to_bin_to_output_magnitude_mad[label] and \
                    predicted_class_to_bin_to_output_magnitude_mad[label][hard_bin] is not None:
                if uncertainty_statistics.UncertaintyStatistics.depth_2_keys_present_in_dictionary(
                        predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin, label, hard_bin):
                    lower_offset = predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin[label][hard_bin]
                else:
                    lower_offset = "NA"
                print(f"\tPredicted label: {label} for hard_q_bin {hard_bin}: Magnitude MAD (only for reference): "
                      f"{predicted_class_to_bin_to_output_magnitude_mad[label][hard_bin]}, "
                      f"Lower offset (for subtraction from rescaled output): {lower_offset}")

    print(f"Embedding summary stats (for normalization): {model.training_embedding_summary_stats}")
    print(f"Estimated class-conditional accuracy over calibration for filtering: "
          f"{model.non_odd_class_conditional_accuracy}")
    test_meta_data, _ = \
        utils_preprocess.get_metadata_lines(options, options.input_eval_set_file,
                                            reduce=False,
                                            use_embeddings=options.use_embeddings,
                                            concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                            calculate_summary_stats=False, is_training=False)
    test_embeddings = test_meta_data["embeddings"].to(main_device)
    test_labels = torch.tensor(test_meta_data["labels"]).to(main_device)
    assert test_embeddings.shape[0] == test_labels.shape[0]
    print(f"test_embeddings.shape: {test_embeddings.shape}")
    test_set_size = test_labels.shape[0]

    q_val_rescaled_by_cdf_by_classConditionalAccuracy = []
    q_val_rescaled_by_cdf_by_classConditionalMeanOutputMagnitude = []
    q_val_rescaled_by_cdf_by_predictionConditionalAccuracy = []
    q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude = []
    hardbin_by_prediction_conditional_sample_sizes = []

    for q in range(constants.default_max_hard_bin):  # This is actually hard-q-bin, rather than the raw q value TODO: update naming
        q_val_rescaled_by_cdf_by_classConditionalAccuracy.append({})
        q_val_rescaled_by_cdf_by_classConditionalMeanOutputMagnitude.append({})
        q_val_rescaled_by_cdf_by_predictionConditionalAccuracy.append({})
        q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude.append({})
        hardbin_by_prediction_conditional_sample_sizes.append({})
        for trueLabel in range(model.numberOfClasses):
            q_val_rescaled_by_cdf_by_classConditionalAccuracy[q][trueLabel] = []
            q_val_rescaled_by_cdf_by_classConditionalMeanOutputMagnitude[q][trueLabel] = []
            q_val_rescaled_by_cdf_by_predictionConditionalAccuracy[q][trueLabel] = []
            q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude[q][trueLabel] = []
            hardbin_by_prediction_conditional_sample_sizes[q][trueLabel] = []

    marginal_accuracy = []
    marginal_accuracy_filtered__lower = []
    marginal_accuracy_filtered__centroid = []
    marginal_accuracy_filtered__upper = []

    class_conditional_accuracy = {}
    class_conditional_accuracy_filtered__lower = {}
    class_conditional_accuracy_filtered__centroid = {}
    class_conditional_accuracy_filtered__upper = {}
    prediction_conditional_accuracy = {}
    prediction_conditional_accuracy_filtered__lower = {}
    prediction_conditional_accuracy_filtered__centroid = {}
    prediction_conditional_accuracy_filtered__upper = {}
    for label in range(model.numberOfClasses):
        class_conditional_accuracy[label] = []
        class_conditional_accuracy_filtered__lower[label] = []
        class_conditional_accuracy_filtered__centroid[label] = []
        class_conditional_accuracy_filtered__upper[label] = []
        prediction_conditional_accuracy[label] = []
        prediction_conditional_accuracy_filtered__lower[label] = []
        prediction_conditional_accuracy_filtered__centroid[label] = []
        prediction_conditional_accuracy_filtered__upper[label] = []

    projected_accuracy_filtered_marginal_original_labels__lower = []
    projected_accuracy_filtered_marginal_original_labels__centroid = []
    projected_accuracy_filtered_marginal_original_labels__upper = []
    # for plotting
    all_prediction_meta_data = []
    # end for plotting
    possible_label_error_json_lines = []
    valid_index_conditional_json_lines = []
    number_of_divisions = 20
    predicted_f_binned = [x for x in range(number_of_divisions)]
    true_frequency_binned = [[] for x in range(number_of_divisions)]

    true_frequency_binned_prediction_conditional = {}
    true_frequency_binned_prediction_conditional__average_sample_sizes = {}
    true_frequency_binned_class_conditional = {}
    for label in range(model.numberOfClasses):
        true_frequency_binned_prediction_conditional[label] = [[] for x in range(number_of_divisions)]
        true_frequency_binned_prediction_conditional__average_sample_sizes[label] = \
            [[] for x in range(number_of_divisions)]
        true_frequency_binned_class_conditional[label] = [[] for x in range(number_of_divisions)]
    instance_i = -1
    for test_embedding, test_label in zip(test_embeddings, test_labels):
        instance_i += 1
        true_test_label = test_label.item()
        prediction_meta_data = \
            model(test_embedding.unsqueeze(0),
                  forward_type=constants.FORWARD_TYPE_SINGLE_PASS_TEST,
                  min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=
                  min_valid_qbin_for_class_conditional_accuracy_with_bounded_error,
                  predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=
                  predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin)

        prediction_meta_data["true_test_label"] = true_test_label  # add true label for plotting functions
        all_prediction_meta_data.append(prediction_meta_data)
        prediction_conditional_distribution__centroid = \
            prediction_meta_data["rescaled_prediction_conditional_distribution__centroid"]
        predicted_class = prediction_meta_data["prediction"]
        prediction_conditional_estimate_of_predicted_class__centroid = \
            prediction_conditional_distribution__centroid[predicted_class].item()

        prediction_conditional_distribution__lower = \
            prediction_meta_data["rescaled_prediction_conditional_distribution__lower"]
        prediction_conditional_estimate_of_predicted_class__lower = \
            prediction_conditional_distribution__lower[predicted_class].item()

        prediction_conditional_distribution__upper = \
            prediction_meta_data["rescaled_prediction_conditional_distribution__upper"]
        prediction_conditional_estimate_of_predicted_class__upper = \
            prediction_conditional_distribution__upper[predicted_class].item()

        hard_qbin = int(prediction_meta_data["soft_qbin__centroid"])
        q_val_rescaled_by_cdf_by_classConditionalAccuracy[hard_qbin][true_test_label].append(
            predicted_class == true_test_label)
        q_val_rescaled_by_cdf_by_predictionConditionalAccuracy[hard_qbin][predicted_class].append(
            predicted_class == true_test_label)
        hardbin_by_prediction_conditional_sample_sizes[hard_qbin][predicted_class].append(
            prediction_meta_data["cumulative_effective_sample_sizes"][predicted_class].item())
        q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude[hard_qbin][predicted_class].append(
            prediction_conditional_estimate_of_predicted_class__centroid)

        marginal_accuracy.append(predicted_class == true_test_label)
        class_conditional_accuracy[true_test_label].append(predicted_class == true_test_label)
        prediction_conditional_accuracy[predicted_class].append(predicted_class == true_test_label)
        if prediction_meta_data["is_valid_index_conditional__lower"]:  # primary quantity of interest
            class_conditional_accuracy_filtered__lower[true_test_label].append(predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__lower[predicted_class].append(predicted_class == true_test_label)
            marginal_accuracy_filtered__lower.append(predicted_class == true_test_label)
            if "original_labels" in test_meta_data and len(test_meta_data["original_labels"]) > 0 \
                    and "original_predictions" in test_meta_data and len(test_meta_data["original_predictions"]) > 0:
                projected_accuracy_filtered_marginal_original_labels__lower.append(
                    test_meta_data["original_labels"][instance_i] == test_meta_data["original_predictions"][instance_i]
                )

            json_obj = {
                "id": test_meta_data['uuids'][instance_i],
                "document": test_meta_data['lines'][instance_i],
                "valid_index__lower": prediction_meta_data["is_valid_index_conditional__lower"],
                "valid_index__centroid": prediction_meta_data["is_valid_index_conditional__centroid"],
                "valid_index__upper": prediction_meta_data["is_valid_index_conditional__upper"],
                "prediction_probability__lower": prediction_conditional_estimate_of_predicted_class__lower,
                "prediction_probability__centroid": prediction_conditional_estimate_of_predicted_class__centroid,
                "prediction_probability__upper": prediction_conditional_estimate_of_predicted_class__upper,
                "prediction": predicted_class,
                "label": true_test_label,
                "n": prediction_meta_data['cumulative_effective_sample_sizes'].detach().cpu().numpy().tolist()
            }
            # first element is for sorting before saving
            valid_index_conditional_json_lines.append(
                (prediction_conditional_estimate_of_predicted_class__lower, json_obj))
            if predicted_class != true_test_label:
                possible_label_error_json_lines.append(
                    (prediction_conditional_estimate_of_predicted_class__lower, json_obj))
        if prediction_meta_data["is_valid_index_conditional__centroid"]:
            class_conditional_accuracy_filtered__centroid[true_test_label].append(predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__centroid[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__centroid.append(predicted_class == true_test_label)
            if predicted_class != true_test_label:
                if "refusals" in test_meta_data and len(test_meta_data["refusals"]) > 0:
                    print(f"///// Valid index conditional (centroid) but the prediction is incorrect. "
                          f"Possible label error:")
                    print(test_meta_data['uuids'][instance_i])
                    print(test_meta_data['lines'][instance_i])
                    print(f'Is valid LOWER: {prediction_meta_data["is_valid_index_conditional__lower"]}')
                    print(f'Is valid UPPER: {prediction_meta_data["is_valid_index_conditional__upper"]}')
                    print(
                        f"\tIs refusal: {test_meta_data['refusals'][instance_i]}, "
                        f"Correctness: {predicted_class == true_test_label}: "
                        f"Predicted class: {predicted_class}; "
                        f"True class: {true_test_label}; "
                        f"Centroid distribution: {prediction_conditional_distribution__centroid}; "
                        f"Sample sizes: {prediction_meta_data['cumulative_effective_sample_sizes']}")
                    print(f"LOWER: {prediction_conditional_estimate_of_predicted_class__lower}; "
                          f"CENTROID: {prediction_conditional_estimate_of_predicted_class__centroid}; "
                          f"UPPER: {prediction_conditional_estimate_of_predicted_class__upper}")
                    print(f"-------END")
                else:
                    print(f"///// Valid index conditional (centroid) but the prediction is incorrect. "
                          f"Possible label error:")
                    print(f'Is valid LOWER: {prediction_meta_data["is_valid_index_conditional__lower"]}')
                    print(f'Is valid UPPER: {prediction_meta_data["is_valid_index_conditional__upper"]}')
                    print(test_meta_data['uuids'][instance_i])
                    print(test_meta_data['lines'][instance_i])
                    print(f"Correctness: {predicted_class == true_test_label}; Predicted class: {predicted_class}; "
                          f"True class: {true_test_label}; "
                          f"Centroid distribution: {prediction_conditional_distribution__centroid}; "
                          f"Sample sizes: {prediction_meta_data['cumulative_effective_sample_sizes']}")
                    print(f"-------END")
            if "original_labels" in test_meta_data and len(test_meta_data["original_labels"]) > 0 \
                    and "original_predictions" in test_meta_data and len(test_meta_data["original_predictions"]) > 0:
                projected_accuracy_filtered_marginal_original_labels__centroid.append(
                    test_meta_data["original_labels"][instance_i] == test_meta_data["original_predictions"][instance_i]
                )
        if prediction_meta_data["is_valid_index_conditional__upper"]:
            class_conditional_accuracy_filtered__upper[true_test_label].append(predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__upper[predicted_class].append(predicted_class == true_test_label)
            marginal_accuracy_filtered__upper.append(predicted_class == true_test_label)
            if "original_labels" in test_meta_data and len(test_meta_data["original_labels"]) > 0 \
                    and "original_predictions" in test_meta_data and len(test_meta_data["original_predictions"]) > 0:
                projected_accuracy_filtered_marginal_original_labels__upper.append(
                    test_meta_data["original_labels"][instance_i] == test_meta_data["original_predictions"][instance_i]
                )
        prediction_conditional_estimate_binned = \
            get_bin(prediction_conditional_estimate_of_predicted_class__centroid, divisions=number_of_divisions)
        true_frequency_binned[prediction_conditional_estimate_binned].append(predicted_class == true_test_label)
        true_frequency_binned_prediction_conditional[predicted_class][prediction_conditional_estimate_binned].append(
            predicted_class == true_test_label)
        true_frequency_binned_prediction_conditional__average_sample_sizes[predicted_class][
            prediction_conditional_estimate_binned].extend(
            prediction_meta_data["cumulative_effective_sample_sizes"].numpy().tolist())
        true_frequency_binned_class_conditional[true_test_label][prediction_conditional_estimate_binned].append(
            predicted_class == true_test_label)

    print(f"######## Conditional estimates ########")
    for label in range(model.numberOfClasses):
        print(f"Label {label} ---")
        print_summary(f"Class-conditional accuracy: Label {label}",
                      class_conditional_accuracy[label])
        print_summary(f"\t>>Class-conditional filtered accuracy LOWER: \t\tLabel {label}",
                      class_conditional_accuracy_filtered__lower[label], total=test_set_size)
        print_summary(f"\t>>Class-conditional filtered accuracy CENTROID: \tLabel {label}",
                      class_conditional_accuracy_filtered__centroid[label], total=test_set_size)
        print_summary(f"\t>>Class-conditional filtered accuracy UPPER: \t\tLabel {label}",
                      class_conditional_accuracy_filtered__upper[label], total=test_set_size)

        print_summary(f"Prediction-conditional accuracy: Label {label}",
                      prediction_conditional_accuracy[label])

        print_summary(f"\t++Prediction-conditional filtered accuracy LOWER: "
                      f"\t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__lower[label], total=test_set_size)
        print_summary(f"\t++Prediction-conditional filtered accuracy CENTROID: \t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__centroid[label], total=test_set_size)
        print_summary(f"\t++Prediction-conditional filtered accuracy UPPER: \t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__upper[label], total=test_set_size)

    print(f"######## Stratified by probability (CENTROID) ########")
    for bin in predicted_f_binned:
        print_summary(f"{bin/number_of_divisions}-{(min(number_of_divisions, bin+1))/number_of_divisions}: "
                      f"PREDICTION CONDITIONAL: Marginal",
                      true_frequency_binned[bin])
        for label in range(model.numberOfClasses):
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

    print(f"######## Stratified by hard q-bin (CENTROID) ########")
    for q in range(constants.default_max_hard_bin):
        for label in range(model.numberOfClasses):
            if len(q_val_rescaled_by_cdf_by_classConditionalAccuracy[q][label]) > 0:
                print(f"hard-q: {q}, label: {label}: class conditional accuracy: \t"
                      f"{np.mean(q_val_rescaled_by_cdf_by_classConditionalAccuracy[q][label])} "
                      f"out of {len(q_val_rescaled_by_cdf_by_classConditionalAccuracy[q][label])})")

            if len(q_val_rescaled_by_cdf_by_predictionConditionalAccuracy[q][label]) > 0:
                print(f"hard-q: {q}, label: {label}: prediction conditional accuracy: \t"
                      f"{np.mean(q_val_rescaled_by_cdf_by_predictionConditionalAccuracy[q][label])} "
                      f"out of {len(q_val_rescaled_by_cdf_by_predictionConditionalAccuracy[q][label])})")
    print(f"######## Stratified by hard q-bin (CENTROID): Additional metrics ########")
    for q in range(constants.default_max_hard_bin):
        for label in range(model.numberOfClasses):
            if len(q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude[q][label]) > 0:
                print(f"hard-q: {q}, label: {label}: prediction conditional mean rescaled output magnitude: "
                      f"min: {np.min(q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude[q][label])} "
                      f"max: {np.max(q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude[q][label])}, "
                      f"mean: {np.mean(q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude[q][label])}, "
                      f"median: {np.median(q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude[q][label])}, "
                      f"out of {len(q_val_rescaled_by_cdf_by_predictionConditionalMeanOutputMagnitude[q][label])})")
            if len(hardbin_by_prediction_conditional_sample_sizes[q][label]) > 0:
                print(
                    f"hard-q: {q}, label: {label}: prediction conditional mean sample size: "
                    f"{np.mean(hardbin_by_prediction_conditional_sample_sizes[q][label])} "
                    f"out of {len(hardbin_by_prediction_conditional_sample_sizes[q][label])})")
    print(f"######## Marginal estimates ########")
    print(f"Marginal accuracy: {np.mean(marginal_accuracy)} out of {len(marginal_accuracy)}")
    # > start temporary LLM reference comparison
    # The projection to original labels is just a reference comparison for cases when a task is encoded such that
    # the LLM generates an answer within the generated text. In practice, the output from sdm() is taken as the final
    # prediction. Typically, for valid index-conditional predictions, there is little to no discrepancy between the LLM
    # prediction and the final prediction from the sdm() activation.
    if len(projected_accuracy_filtered_marginal_original_labels__lower) > 0:
        print_summary(f"Filtered valid marginal, projected to original task labels LOWER:",
                      projected_accuracy_filtered_marginal_original_labels__lower)
    if len(projected_accuracy_filtered_marginal_original_labels__centroid) > 0:
        print_summary(f"Filtered valid marginal, projected to original task labels CENTROID:",
                      projected_accuracy_filtered_marginal_original_labels__centroid)
    if len(projected_accuracy_filtered_marginal_original_labels__upper) > 0:
        print_summary(f"Filtered valid marginal, projected to original task labels UPPER:",
                      projected_accuracy_filtered_marginal_original_labels__upper)
    # > end temporary LLM reference comparison
    print(
        f"Filtered valid marginal (filtered by valid index conditional) LOWER: "
        f"{np.mean(marginal_accuracy_filtered__lower)} out of {len(marginal_accuracy_filtered__lower)} "
        f"({len(marginal_accuracy_filtered__lower)/len(marginal_accuracy)})")
    print(
        f"Filtered valid marginal (filtered by valid index conditional) CENTROID: "
        f"{np.mean(marginal_accuracy_filtered__centroid)} out of {len(marginal_accuracy_filtered__centroid)} "
        f"({len(marginal_accuracy_filtered__centroid)/len(marginal_accuracy)})")
    print(
            f"Filtered valid marginal (filtered by valid index conditional) UPPER: "
            f"{np.mean(marginal_accuracy_filtered__upper)} out of {len(marginal_accuracy_filtered__upper)} "
            f"({len(marginal_accuracy_filtered__upper)/len(marginal_accuracy)})")

    possible_label_error_json_lines = [y[1] for y in sorted(possible_label_error_json_lines, key=lambda x: x[0],
                                                            reverse=True)]
    if options.label_error_file != "" and len(possible_label_error_json_lines) > 0:
        utils_model.save_json_lines(options.label_error_file, possible_label_error_json_lines)
        print(f"{len(possible_label_error_json_lines)} candidate label errors saved to {options.label_error_file}")

    valid_index_conditional_json_lines = [y[1] for y in sorted(valid_index_conditional_json_lines, key=lambda x: x[0],
                                                            reverse=True)]
    if options.valid_index_conditional_file != "" and len(valid_index_conditional_json_lines) > 0:
        utils_model.save_json_lines(options.valid_index_conditional_file, valid_index_conditional_json_lines)
        print(f"{len(valid_index_conditional_json_lines)} valid index-conditional predictions saved to {options.valid_index_conditional_file}")


def test_gen_ai(options, main_device, gen_ai_model, tokenizer, input_eval_set_file, llmType):
    model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"), load_for_inference=True)
    # global_uncertainty_statistics = utils_model.load_global_uncertainty_statistics_from_disk(options.model_dir)

    calibration_count_accepted_among_valid_class_conditional = \
        utils_train_main_gen_ai_router.compute_generation_sdm_for_eval(
            options, model, main_device, gen_ai_model, tokenizer, input_eval_set_file,
            llmType,
            generation_output_label="eval_run_",
            load_final_llm_weights=True)

