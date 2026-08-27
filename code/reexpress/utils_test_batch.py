# Copyright Reexpress AI, Inc. All rights reserved.
import copy
import os

import torch

import numpy as np
import constants
import utils_model
import sdm_model
import utils_preprocess
import data_validator
import utils_eval_batch

from utils_latex import print_latex_row, init_latex_rows_dict, get_acc_prop_tuple, MARGINAL_ACC_KEY, \
    MARGINAL_ADMITTED_KEY, CLASS_CONDITIONAL_ACC_KEY, CLASS_CONDITIONAL_ADMITTED_KEY, PREDICTION_CONDITIONAL_ACC_KEY, \
    PREDICTION_CONDITIONAL_ADMITTED_KEY


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


def get_accuracy_cell(outcomes):
    # outcomes is a list of booleans; the rendered cell is "accuracy (count)", with "--" for an empty list:
    if len(outcomes) == 0:
        return "-- (0)"
    return f"{np.mean(outcomes):.4f} ({len(outcomes)})"


def format_ascii_table(header_row, rows):
    # Renders an aligned ASCII table: the first column is left-aligned; all other columns are right-aligned.
    all_rows = [header_row] + rows
    column_widths = [max(len(str(row[column_i])) for row in all_rows) for column_i in range(len(header_row))]

    def format_row(row):
        formatted_cells = []
        for column_i, cell in enumerate(row):
            cell = str(cell)
            formatted_cells.append(cell.ljust(column_widths[column_i]) if column_i == 0
                                   else cell.rjust(column_widths[column_i]))
        return " | ".join(formatted_cells)
    lines = [format_row(header_row), "-+-".join("-" * column_width for column_width in column_widths)]
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def print_per_region_conditional_accuracy_table(table_label, hr_regions, per_region_outcomes, numberOfClasses):
    # Prints the class- (i.e., true-label-) conditional, prediction-conditional, and marginal accuracy of the
    # eval instances assigned to each nested high-reliability region, with the final row ("none (0.)")
    # collecting the instances not assigned to any region (which the caller treats as maximally uncertain).
    # The rows partition the evaluated instances.
    print(table_label)
    print(f"    Cells: accuracy (n). y=c conditions on the true label; pred=c conditions on the predicted "
          f"label.")
    header_row = ["alpha", "marginal"]
    header_row.extend([f"y={label}" for label in range(numberOfClasses)])
    header_row.extend([f"pred={label}" for label in range(numberOfClasses)])
    rows = []
    for region_alpha in [hr_region["alpha"] for hr_region in hr_regions] + [0.0]:
        outcomes = per_region_outcomes[region_alpha]
        row = ["none (0.)" if region_alpha == 0.0 else f"{region_alpha}"]
        row.append(get_accuracy_cell(outcomes["marginal"]))
        row.extend([get_accuracy_cell(outcomes["by_true"][label]) for label in range(numberOfClasses)])
        row.extend([get_accuracy_cell(outcomes["by_pred"][label]) for label in range(numberOfClasses)])
        rows.append(row)
    print(format_ascii_table(header_row, rows))


def get_accuracy_and_proportion_cell(outcomes, total_instances):
    # outcomes is a list of booleans; the rendered cell is "accuracy (percent_of_total%=count)", where the
    # percentage is out of the total evaluated instances, with "--" for an empty list:
    if len(outcomes) == 0:
        return f"-- (0.0%=0)"
    return f"{np.mean(outcomes):.4f} ({100 * len(outcomes) / total_instances:.1f}%={len(outcomes)})"


def print_cumulative_region_conditional_accuracy_table(table_label, hr_regions, per_region_outcomes,
                                                       numberOfClasses, total_instances):
    # The cumulative counterpart to print_per_region_conditional_accuracy_table(): each row "alpha >= x" is
    # the union of the instances assigned to the regions with alpha >= x (i.e., unions of rungs down the
    # ladder), which is the operating-point view for a caller that accepts any region at or above a given
    # alpha. The parenthetical percentage in each cell is the count as a proportion of the total evaluated
    # instances, so within a row, the y=c (and, respectively, pred=c) percentages sum to the marginal
    # percentage. The complement of the final row is the set of instances not assigned to any region.
    print(table_label)
    print(f"    Cells: accuracy (percent of the {total_instances} evaluated instances=n). y=c conditions on "
          f"the true label; pred=c conditions on the predicted label.")
    if len(hr_regions) == 0:
        print(f"    (No recorded regions.)")
        return
    header_row = ["alpha", "marginal"]
    header_row.extend([f"y={label}" for label in range(numberOfClasses)])
    header_row.extend([f"pred={label}" for label in range(numberOfClasses)])
    cumulative_outcomes = {"marginal": [],
                           "by_true": {label: [] for label in range(numberOfClasses)},
                           "by_pred": {label: [] for label in range(numberOfClasses)}}
    rows = []
    for hr_region in hr_regions:
        region_outcomes = per_region_outcomes[hr_region["alpha"]]
        cumulative_outcomes["marginal"].extend(region_outcomes["marginal"])
        for label in range(numberOfClasses):
            cumulative_outcomes["by_true"][label].extend(region_outcomes["by_true"][label])
            cumulative_outcomes["by_pred"][label].extend(region_outcomes["by_pred"][label])
        row = [f">= {hr_region['alpha']}"]
        row.append(get_accuracy_and_proportion_cell(cumulative_outcomes["marginal"], total_instances))
        row.extend([get_accuracy_and_proportion_cell(cumulative_outcomes["by_true"][label], total_instances)
                    for label in range(numberOfClasses)])
        row.extend([get_accuracy_and_proportion_cell(cumulative_outcomes["by_pred"][label], total_instances)
                    for label in range(numberOfClasses)])
        rows.append(row)
    print(format_ascii_table(header_row, rows))


def test(options, main_device, iteration_model_dir=None, id2ensemble_stats=None):
    print(f"###############Beginning of evaluation###############")
    if iteration_model_dir is None:
        model = \
            utils_model.load_model_torch(options.model_dir, main_device, load_for_inference=True)
    else:
        assert id2ensemble_stats is not None
        model = \
            utils_model.load_model_torch(iteration_model_dir, main_device, load_for_inference=True)
    # global uncertainty statistics are always loaded from the main model_dir. These are informational and not
    # used directly in the calculations below.
    global_uncertainty_statistics = utils_model.load_global_uncertainty_statistics_from_disk(options.model_dir)

    if model.calibration_training_stage == sdm_model.modelCalibrationTrainingStages.init:
        print(f"The model has not been trained. Exiting.")
        exit()
    if model.calibration_training_stage != sdm_model.modelCalibrationTrainingStages.complete:
        print(f"The model has not been calibrated. "
              f"Train from scratch or run --options.recalibrate_with_updated_alpha. Exiting.")
        exit()

    if model.alpha_resolution != options.alpha_resolution:
        print(f"The alpha_resolution used for calibration was {model.alpha_resolution}, but "
              f"{options.alpha_resolution} was requested. "
              f"Run --options.recalibrate_with_updated_alpha_resolution to recalibrate based on the new value. "
              f"Exiting.")
        exit()

    if options.is_training_support:
        assert model.is_sdm_network_verification_layer, \
            "The training set empirical CDFs for the " \
            "distances are only saved if --is_sdm_network_verification_layer to save space."
        print("--options.is_training_support was provided, so the calculations of Similarity will assume the "
              "first match is identity.")

    global_uncertainty_statistics.validate_min_rescaled_similarities()

    hr_region_stats = model.get_most_conservative_high_reliability_region_stats()
    alpha_prime = hr_region_stats["most_conservative_hr_alpha"]
    print(f"For this chosen model iteration:")
    print(f'\tMost conservative alpha to determine the high reliability region: {alpha_prime}, '
          f'\n\tCorresponding q\'_min: {hr_region_stats["most_conservative_hr_min_rescaled_similarity"]},'
          f'\n\tCorresponding class-wise thresholds (psi): {hr_region_stats["most_conservative_hr_output_thresholds"]}')

    print(f"Embedding summary stats (for normalization): {model.training_embedding_summary_stats}")

    if getattr(options, "eval_on_best_iteration_calibration_split", False):
        # Evaluate on the calibration split of the best training iteration, reconstructed from the split
        # indices saved during training with converted dataset directories as input
        # (--input_eval_set_file is ignored in this case):
        print(f"Evaluating on the best-iteration calibration split reconstructed from "
              f"{os.path.join(options.model_dir, 'best_iteration_data')}")
        test_meta_data = utils_preprocess.load_best_iteration_calibration_split(options.model_dir)
    else:
        test_meta_data, _ = \
            utils_preprocess.get_metadata_lines(options, options.input_eval_set_file,
                                                reduce=False,
                                                use_embeddings=options.use_embeddings,
                                                concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                calculate_summary_stats=False, is_training=False)
    test_embeddings = test_meta_data["embeddings"]
    test_labels = torch.tensor(test_meta_data["labels"])
    assert test_embeddings.shape[0] == test_labels.shape[0]
    print(f"test_embeddings.shape: {test_embeddings.shape}")
    test_set_size = test_labels.shape[0]

    # BEGIN process batch
    test_dataset_q_values, test_dataset_distance_quantile_per_class, test_batch_f_outputs, \
        test_d0_values, test_nearest_support_idx = \
        utils_eval_batch.get_q_and_d_from_embeddings(model, eval_batch_size=test_set_size,
                                                     eval_embeddings=test_embeddings,
                                                     main_device=main_device,
                                                     is_training_support=options.is_training_support)
    test_per_class_loss_as_list, test_balanced_loss, test_marginal_loss, \
        test_per_class_accuracy_as_list, test_balanced_accuracy, test_marginal_accuracy, \
        test_per_class_q_as_list, test_balanced_q, test_marginal_q, \
        test_sdm_outputs = \
        utils_eval_batch.get_metrics_from_cached_outputs(test_set_size, model,
                                                         test_batch_f_outputs,
                                                         main_device,
                                                         test_labels,
                                                         q_values=test_dataset_q_values,
                                                         distance_quantile_per_class=test_dataset_distance_quantile_per_class)
    rescaled_similarities, predictions = \
        model.get_rescaled_similarity_for_eval_batch(
            cached_f_outputs=test_batch_f_outputs,
            dataset_q_values=test_dataset_q_values,
            sdm_outputs=test_sdm_outputs)

    utils_eval_batch.print_metrics(e=-1, numberOfClasses=model.numberOfClasses, split_label_name="Eval set",
                                   per_class_loss_as_list=test_per_class_loss_as_list,
                                   balanced_loss=test_balanced_loss,
                                   marginal_loss=test_marginal_loss,
                                   per_class_accuracy_as_list=test_per_class_accuracy_as_list,
                                   balanced_accuracy=test_balanced_accuracy,
                                   marginal_accuracy=test_marginal_accuracy,
                                   per_class_q_as_list=test_per_class_q_as_list,
                                   balanced_q=test_balanced_q,
                                   marginal_q=test_marginal_q)
    # END process batch

    results = model.get_batch_eval_output_dictionary(
        rescaled_similarities=rescaled_similarities,
        sdm_batch_outputs=test_sdm_outputs,
        predictions=predictions,
        batch_f=test_batch_f_outputs.to(main_device),
        batch_q=test_dataset_q_values.to(main_device),
        batch_distance_quantile_per_class=test_dataset_distance_quantile_per_class.to(main_device),
        d0_values=test_d0_values,
        nearest_support_idx_values=test_nearest_support_idx)

    latex_rows_dict_no_reject = init_latex_rows_dict(numberOfClasses=model.numberOfClasses)
    latex_rows_dict_softmax_f = init_latex_rows_dict(numberOfClasses=model.numberOfClasses)
    latex_rows_dict_softmax_df = init_latex_rows_dict(numberOfClasses=model.numberOfClasses)
    latex_rows_dict_sdm = init_latex_rows_dict(numberOfClasses=model.numberOfClasses)
    latex_rows_dict_sdm_hr = init_latex_rows_dict(numberOfClasses=model.numberOfClasses)
    latex_rows_dict_sdm_hr_lower = init_latex_rows_dict(numberOfClasses=model.numberOfClasses)

    q_val_rescaled_by_sdm_by_classConditionalAccuracy = []
    q_val_rescaled_by_sdm_by_predictionConditionalAccuracy = []

    for q in range(model.maxQAvailableFromIndexer+1):
        q_val_rescaled_by_sdm_by_classConditionalAccuracy.append({})
        q_val_rescaled_by_sdm_by_predictionConditionalAccuracy.append({})
        for trueLabel in range(model.numberOfClasses):
            q_val_rescaled_by_sdm_by_classConditionalAccuracy[q][trueLabel] = []
            q_val_rescaled_by_sdm_by_predictionConditionalAccuracy[q][trueLabel] = []

    marginal_accuracy = []
    marginal_accuracy_filtered__sdm_by_hr_region = []
    marginal_accuracy_filtered__sdm_by_hr_region_lower = []
    marginal_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime = []
    marginal_accuracy_filtered__softmax_of_f_by_alpha_prime = []
    marginal_accuracy_filtered__sdm_by_alpha_prime = []

    class_conditional_accuracy = {}
    class_conditional_accuracy_filtered__sdm_by_hr_region = {}
    class_conditional_accuracy_filtered__sdm_by_hr_region_lower = {}
    class_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime = {}
    class_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime = {}
    class_conditional_accuracy_filtered__sdm_by_alpha_prime = {}

    class_conditional_accuracy__is_ood_sdm = {}
    class_conditional_accuracy__NOT_is_ood__AND__NOT_is_hr_region = {}

    prediction_conditional_accuracy = {}
    prediction_conditional_accuracy_filtered__sdm_by_hr_region = {}
    prediction_conditional_accuracy_filtered__sdm_by_hr_region_lower = {}
    prediction_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime = {}
    prediction_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime = {}
    prediction_conditional_accuracy_filtered__sdm_by_alpha_prime = {}
    for label in range(model.numberOfClasses):
        class_conditional_accuracy[label] = []
        class_conditional_accuracy_filtered__sdm_by_hr_region[label] = []
        class_conditional_accuracy_filtered__sdm_by_hr_region_lower[label] = []
        class_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime[label] = []
        class_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime[label] = []
        class_conditional_accuracy_filtered__sdm_by_alpha_prime[label] = []
        class_conditional_accuracy__is_ood_sdm[label] = []
        class_conditional_accuracy__NOT_is_ood__AND__NOT_is_hr_region[label] = []

        prediction_conditional_accuracy[label] = []
        prediction_conditional_accuracy_filtered__sdm_by_hr_region[label] = []
        prediction_conditional_accuracy_filtered__sdm_by_hr_region_lower[label] = []
        prediction_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime[label] = []
        prediction_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime[label] = []
        prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[label] = []

    # Per-region (nested alpha ladder) conditional accuracy summary stats, for the tables printed at the end
    # of evaluation. The keys are the exact region alpha values (which are exactly representable, so the
    # per-instance "hr_region_alpha"/"hr_region_alpha_lower" values match exactly), with the key 0.0
    # collecting the instances not assigned to any region:
    per_region_outcomes__sdm_by_hr_region = {}
    per_region_outcomes__sdm_by_hr_region_lower = {}
    for region_alpha in [hr_region["alpha"] for hr_region in model.hr_regions] + [0.0]:
        for per_region_outcomes in (per_region_outcomes__sdm_by_hr_region,
                                    per_region_outcomes__sdm_by_hr_region_lower):
            per_region_outcomes[region_alpha] = {
                "marginal": [],
                "by_true": {label: [] for label in range(model.numberOfClasses)},
                "by_pred": {label: [] for label in range(model.numberOfClasses)}}

    # for plotting
    # all_prediction_meta_data = []
    # end for plotting
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
    for label in range(model.numberOfClasses):
        true_frequency_binned_prediction_conditional[label] = [[] for x in range(number_of_divisions)]
        true_frequency_binned_prediction_conditional__average_sample_sizes[label] = \
            [[] for x in range(number_of_divisions)]
        true_frequency_binned_class_conditional[label] = [[] for x in range(number_of_divisions)]
    instance_i = -1
    number_of_unlabeled_labels = 0
    number_of_ood_labels = 0

    # Another pass to collect additional metrics and to construct the output JSON (and optionally, LaTex rows)
    for prediction_meta_data, test_label in zip(results, test_labels):

        instance_i += 1
        true_test_label = test_label.item()
        assert data_validator.isValidLabel(label=true_test_label, numberOfClasses=model.numberOfClasses)
        if not data_validator.isKnownValidLabel(label=true_test_label, numberOfClasses=model.numberOfClasses):
            if true_test_label == data_validator.unlabeledLabel:
                number_of_unlabeled_labels += 1
            elif true_test_label == data_validator.oodLabel:
                number_of_ood_labels += 1
            continue

        # prediction_meta_data["true_test_label"] = true_test_label
        # all_prediction_meta_data.append(prediction_meta_data)
        predicted_class = prediction_meta_data["prediction"]

        prediction_conditional_distribution__sdm = \
            prediction_meta_data["sdm_output"]

        prediction_conditional_estimate_of_predicted_class__sdm = \
            prediction_conditional_distribution__sdm[predicted_class].item()

        prediction_conditional_estimate_of_predicted_class__sdm_lower = \
            prediction_meta_data["sdm_output_d_lower"][predicted_class].item()  # using DKW

        prediction_conditional_distribution__softmax_of_d_times_f = \
            torch.softmax(prediction_meta_data["d"] * prediction_meta_data["f"], dim=-1)
        prediction_conditional_estimate_of_predicted_class__softmax_of_d_times_f = \
            prediction_conditional_distribution__softmax_of_d_times_f[predicted_class].item()

        prediction_conditional_distribution__softmax_of_f = \
            torch.softmax(prediction_meta_data["f"], dim=-1)
        prediction_conditional_estimate_of_predicted_class__softmax_of_f = \
            prediction_conditional_distribution__softmax_of_f[predicted_class].item()

        floor_rescaled_similarity = prediction_meta_data["floor_rescaled_similarity"]
        q_val_rescaled_by_sdm_by_classConditionalAccuracy[floor_rescaled_similarity][true_test_label].append(
            predicted_class == true_test_label)
        q_val_rescaled_by_sdm_by_predictionConditionalAccuracy[floor_rescaled_similarity][predicted_class].append(
            predicted_class == true_test_label)

        marginal_accuracy.append(predicted_class == true_test_label)
        class_conditional_accuracy[true_test_label].append(predicted_class == true_test_label)
        prediction_conditional_accuracy[predicted_class].append(predicted_class == true_test_label)

        # Per-region (nested alpha ladder) summary stats, keyed by the exact assigned region alpha
        # (0.0 for instances not assigned to any region):
        for assigned_region_alpha, per_region_outcomes in (
                (prediction_meta_data["hr_region_alpha"], per_region_outcomes__sdm_by_hr_region),
                (prediction_meta_data["hr_region_alpha_lower"], per_region_outcomes__sdm_by_hr_region_lower)):
            per_region_outcomes[assigned_region_alpha]["marginal"].append(predicted_class == true_test_label)
            per_region_outcomes[assigned_region_alpha]["by_true"][true_test_label].append(
                predicted_class == true_test_label)
            per_region_outcomes[assigned_region_alpha]["by_pred"][predicted_class].append(
                predicted_class == true_test_label)

        if prediction_meta_data["is_ood"]:
            class_conditional_accuracy__is_ood_sdm[true_test_label].append(predicted_class == true_test_label)
        if not prediction_meta_data["is_ood"] and not prediction_meta_data["is_high_reliability_region"]:
            class_conditional_accuracy__NOT_is_ood__AND__NOT_is_hr_region[true_test_label].append(
                predicted_class == true_test_label)
        json_obj = {}
        if prediction_meta_data["is_high_reliability_region"] or options.prediction_output_file != "" \
                or id2ensemble_stats is not None:
            json_obj = copy.deepcopy(prediction_meta_data)
            json_obj[constants.REEXPRESS_ID_KEY] = test_meta_data['uuids'][instance_i]
            json_obj[constants.REEXPRESS_DOCUMENT_KEY] = test_meta_data['lines'][instance_i]
            json_obj[constants.REEXPRESS_LABEL_KEY] = true_test_label
            # Switch to python data structures for saving to JSON:
            json_obj["cumulative_effective_sample_sizes"] = \
                prediction_meta_data['cumulative_effective_sample_sizes'].detach().cpu().numpy().tolist()
            json_obj["f"] = \
                prediction_meta_data['f'].detach().cpu().numpy().tolist()
            json_obj["sdm_output"] = \
                prediction_meta_data['sdm_output'].detach().cpu().numpy().tolist()
            json_obj["sdm_output_d_lower"] = \
                prediction_meta_data['sdm_output_d_lower'].detach().cpu().numpy().tolist()
            json_obj["sdm_output_d_upper"] = \
                prediction_meta_data['sdm_output_d_upper'].detach().cpu().numpy().tolist()
            # convenience for constructing/analyzing ensembles:
            json_obj["hr_region_alpha"] = prediction_meta_data["hr_region_alpha"]
            json_obj["hr_region_alpha_lower"] = prediction_meta_data["hr_region_alpha_lower"]

        if prediction_meta_data["is_high_reliability_region"]:  # HR region
            class_conditional_accuracy_filtered__sdm_by_hr_region[true_test_label].append(
                predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__sdm_by_hr_region[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__sdm_by_hr_region.append(predicted_class == true_test_label)
            # first two elements are for sorting before saving
            predictions_in_high_reliability_region_json_lines.append(
                (prediction_conditional_estimate_of_predicted_class__sdm,
                 prediction_meta_data["rescaled_similarity"],
                 json_obj))
            if predicted_class != true_test_label:
                possible_label_error_json_lines.append(
                    (prediction_conditional_estimate_of_predicted_class__sdm,
                     prediction_meta_data["rescaled_similarity"],
                     json_obj))
        if prediction_meta_data["is_high_reliability_region_lower"]:  # HR_lower region
            class_conditional_accuracy_filtered__sdm_by_hr_region_lower[true_test_label].append(
                predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__sdm_by_hr_region_lower[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__sdm_by_hr_region_lower.append(predicted_class == true_test_label)
            # first two elements are for sorting before saving
            predictions_in_high_reliability_region_json_lines_lower.append(
                (prediction_conditional_estimate_of_predicted_class__sdm_lower,
                 prediction_meta_data["rescaled_similarity_lower"],
                 json_obj))
            if predicted_class != true_test_label:
                possible_label_error_json_lines_lower.append(
                    (prediction_conditional_estimate_of_predicted_class__sdm_lower,
                     prediction_meta_data["rescaled_similarity_lower"],
                     json_obj))

        if options.prediction_output_file != "":  # for saving all lines
            all_predictions_json_lines.append(json_obj)

        if id2ensemble_stats is not None:
            if json_obj[constants.REEXPRESS_ID_KEY] in id2ensemble_stats:
                id2ensemble_stats[json_obj[constants.REEXPRESS_ID_KEY]].append(json_obj)
            else:
                id2ensemble_stats[json_obj[constants.REEXPRESS_ID_KEY]] = [json_obj]
        if prediction_conditional_estimate_of_predicted_class__sdm >= alpha_prime:
            class_conditional_accuracy_filtered__sdm_by_alpha_prime[true_test_label].append(predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__sdm_by_alpha_prime.append(predicted_class == true_test_label)

        if prediction_conditional_estimate_of_predicted_class__softmax_of_d_times_f >= alpha_prime:
            class_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime[true_test_label].append(predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime.append(predicted_class == true_test_label)
        if prediction_conditional_estimate_of_predicted_class__softmax_of_f >= alpha_prime:
            class_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime[true_test_label].append(predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime[predicted_class].append(predicted_class == true_test_label)
            marginal_accuracy_filtered__softmax_of_f_by_alpha_prime.append(predicted_class == true_test_label)

        prediction_conditional_estimate_binned = \
            get_bin(prediction_conditional_estimate_of_predicted_class__sdm, divisions=number_of_divisions)
        true_frequency_binned[prediction_conditional_estimate_binned].append(predicted_class == true_test_label)
        true_frequency_binned_prediction_conditional[predicted_class][prediction_conditional_estimate_binned].append(
            predicted_class == true_test_label)
        true_frequency_binned_prediction_conditional__average_sample_sizes[predicted_class][
            prediction_conditional_estimate_binned].extend(
            prediction_meta_data["cumulative_effective_sample_sizes"].detach().cpu().numpy().tolist())
        true_frequency_binned_class_conditional[true_test_label][prediction_conditional_estimate_binned].append(
            predicted_class == true_test_label)

    print(f"######## Conditional estimates ########")
    print(f"\tLegend: 'HR': High Reliability region")
    for label in range(model.numberOfClasses):
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
    for label in range(model.numberOfClasses):
        print(f"Label {label} ---")
        print_summary(f"Class-conditional accuracy (not HR AND not OOD): Label {label}",
                      class_conditional_accuracy__NOT_is_ood__AND__NOT_is_hr_region[label],
                      total=test_set_size)
    print(f"######## Class-Conditional estimates for OOD ########")
    for label in range(model.numberOfClasses):
        print(f"Label {label} ---")
        print_summary(f"Class-conditional accuracy (OOD): Label {label}",
                      class_conditional_accuracy__is_ood_sdm[label],
                      total=test_set_size)

    print(f"######## Additional reference conditional estimates ########")
    for label in range(model.numberOfClasses):
        print(f"Label {label} ---")
        print_summary(f"\t-df-Ref: Class-conditional accuracy (softmax(d*f)_predicted >= {alpha_prime}): \t\tLabel {label}",
                      class_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime[label],
                      total=test_set_size)
        latex_rows_dict_softmax_df[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_softmax_df[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
                get_acc_prop_tuple(class_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime[label],
                                   total=test_set_size)

        print_summary(f"\t-f-Ref: Class-conditional accuracy (softmax(f)_predicted >= {alpha_prime}): "
                      f"\t\tLabel {label}",
                      class_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime[label], total=test_set_size)
        latex_rows_dict_softmax_f[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_softmax_f[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime[label],
                               total=test_set_size)
        print_summary(f"\t-df-Ref: Prediction-conditional accuracy (softmax(d*f)_predicted >= {alpha_prime}): \t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime[label],
                      total=test_set_size)
        latex_rows_dict_softmax_df[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_softmax_df[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime[label],
                               total=test_set_size)
        print_summary(f"\t-f-Ref: Prediction-conditional accuracy (softmax(f)_predicted >= {alpha_prime}): "
                      f"\t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime[label], total=test_set_size)
        latex_rows_dict_softmax_f[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_softmax_f[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy_filtered__softmax_of_f_by_alpha_prime[label],
                               total=test_set_size)

    print(f"######## Stratified by probability, sdm(z') ########")
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

    print(f"######## Stratified by floor of the rescaled Similarity (q') ########")
    for q in range(model.maxQAvailableFromIndexer+1):
        for label in range(model.numberOfClasses):
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

        print(f">>Additional reference marginal estimates<<")
        print(
            f"-df-Ref: Filtered marginal (constrained to softmax(d*f)_predicted >= {alpha_prime}): "
            f"{np.mean(marginal_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime)} out of "
            f"{len(marginal_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime)} "
            f"({len(marginal_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime)/len(marginal_accuracy)})")
        latex_rows_dict_softmax_df[MARGINAL_ACC_KEY], \
            latex_rows_dict_softmax_df[MARGINAL_ADMITTED_KEY] = \
            get_acc_prop_tuple(marginal_accuracy_filtered__softmax_of_d_times_f_by_alpha_prime,
                               total=len(marginal_accuracy))

        print(
            f"-f-Ref: Filtered marginal (constrained to softmax(f)_predicted >= {alpha_prime}): "
            f"{np.mean(marginal_accuracy_filtered__softmax_of_f_by_alpha_prime)} out of "
            f"{len(marginal_accuracy_filtered__softmax_of_f_by_alpha_prime)} "
            f"({len(marginal_accuracy_filtered__softmax_of_f_by_alpha_prime)/len(marginal_accuracy)})")
        latex_rows_dict_softmax_f[MARGINAL_ACC_KEY], \
            latex_rows_dict_softmax_f[MARGINAL_ADMITTED_KEY] = \
            get_acc_prop_tuple(marginal_accuracy_filtered__softmax_of_f_by_alpha_prime, total=len(marginal_accuracy))

    print(f"######## OOD/Unlabeled stats ########")
    print(f"Count of unlabeled labeled (i.e., label=={data_validator.unlabeledLabel}) "
          f"instances (ignored above): {number_of_unlabeled_labels} out of {test_set_size}")
    print(f"Count of OOD labeled (i.e., label=={data_validator.oodLabel}) "
          f"instances (ignored above): {number_of_ood_labels} out of {test_set_size}")
    print(f"######## ########")
    if id2ensemble_stats is None:  # don't save for ensemble runs, because this will overwrite the files
        # Save error and admitted predictions for HR_lower
        possible_label_error_json_lines_lower = [y[2] for y in sorted(possible_label_error_json_lines_lower,
                                                                      key=lambda x: (x[0], x[1]),
                                                                      reverse=True)]
        if options.label_error_hr_lower_file != "" and len(possible_label_error_json_lines_lower) > 0:
            utils_model.save_json_lines(options.label_error_hr_lower_file, possible_label_error_json_lines_lower)
            print(f">{len(possible_label_error_json_lines_lower)} candidate label errors "
                  f"(in HR_lower but y != prediction) saved to {options.label_error_hr_lower_file}")

        predictions_in_high_reliability_region_json_lines_lower = \
            [y[2] for y in sorted(predictions_in_high_reliability_region_json_lines_lower,
                                  key=lambda x: (x[0], x[1]),
                                  reverse=True)]
        if options.predictions_in_high_reliability_region_lower_file != "" and \
                len(predictions_in_high_reliability_region_json_lines_lower) > 0:
            utils_model.save_json_lines(options.predictions_in_high_reliability_region_lower_file,
                                        predictions_in_high_reliability_region_json_lines_lower)
            print(f">{len(predictions_in_high_reliability_region_json_lines_lower)} "
                  f"high reliability predictions (in HR_lower) saved to "
                  f"{options.predictions_in_high_reliability_region_lower_file}")

        # Additionally save error and admitted predictions for HR (centroid)
        possible_label_error_json_lines = [y[2] for y in sorted(possible_label_error_json_lines,
                                                                key=lambda x: (x[0], x[1]),
                                                                reverse=True)]
        if options.label_error_file != "" and len(possible_label_error_json_lines) > 0:
            utils_model.save_json_lines(options.label_error_file, possible_label_error_json_lines)
            print(f">{len(possible_label_error_json_lines)} candidate label errors "
                  f"(in HR but y != prediction) saved to {options.label_error_file}")

        predictions_in_high_reliability_region_json_lines = \
            [y[2] for y in sorted(predictions_in_high_reliability_region_json_lines,
                                  key=lambda x: (x[0], x[1]),
                                  reverse=True)]
        if options.predictions_in_high_reliability_region_file != "" and \
                len(predictions_in_high_reliability_region_json_lines) > 0:
            utils_model.save_json_lines(options.predictions_in_high_reliability_region_file,
                                        predictions_in_high_reliability_region_json_lines)
            print(f">{len(predictions_in_high_reliability_region_json_lines)} "
                  f"high reliability predictions (in HR) saved to "
                  f"{options.predictions_in_high_reliability_region_file}")

        # All predictions file:
        if options.prediction_output_file != "" and len(all_predictions_json_lines) > 0:
            utils_model.save_json_lines(options.prediction_output_file, all_predictions_json_lines)
            print(f">The prediction for each document (total: {len(all_predictions_json_lines)}) has been saved to "
                  f"{options.prediction_output_file}")

    assert test_set_size == instance_i + 1, "ERROR: The index is mismatched."
    print_latex_row(options, model, alpha_prime,
                    latex_rows_dict_no_reject, latex_rows_dict_softmax_f,
                    latex_rows_dict_softmax_df, latex_rows_dict_sdm, latex_rows_dict_sdm_hr,
                    latex_rows_dict_sdm_hr_lower)

    for hr_region in model.hr_regions:
        print(hr_region)

    print("")
    print_per_region_conditional_accuracy_table(
        'Per-region conditional accuracy: HR region assignment ("hr_region_alpha")',
        model.hr_regions, per_region_outcomes__sdm_by_hr_region, model.numberOfClasses)
    print("")
    print_per_region_conditional_accuracy_table(
        'Per-region conditional accuracy: HR^{lower} region assignment ("hr_region_alpha_lower")',
        model.hr_regions, per_region_outcomes__sdm_by_hr_region_lower, model.numberOfClasses)
    print("")
    print_cumulative_region_conditional_accuracy_table(
        'Cumulative conditional accuracy: HR region assignment ("hr_region_alpha")',
        model.hr_regions, per_region_outcomes__sdm_by_hr_region, model.numberOfClasses, test_set_size)
    print("")
    print_cumulative_region_conditional_accuracy_table(
        'Cumulative conditional accuracy: HR^{lower} region assignment ("hr_region_alpha_lower")',
        model.hr_regions, per_region_outcomes__sdm_by_hr_region_lower, model.numberOfClasses, test_set_size)

    return id2ensemble_stats
