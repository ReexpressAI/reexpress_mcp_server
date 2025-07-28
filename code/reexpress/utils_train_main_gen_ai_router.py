# Copyright Reexpress AI, Inc. All rights reserved.

from sdm_model import SimilarityDistanceMagnitudeCalibrator
import constants
import utils_classification
import utils_model
# import utils_gen
import data_validator
from utils_test import print_summary

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import argparse
import copy

import math
import logging
import sys
from os import path
import time


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def set_genai_gradient_update_state_inplace(model, requires_grad=True):
    # print(f"Training negative and positive distributions.")
    # model.fc_negative.weight.requires_grad = requires_grad
    print(f"FREEZING NEGATIVE/ORIGINAL LLM DISTRIBUTION WEIGHTS")
    model.fc_negative.weight.requires_grad = False

    model.fc_positive.weight.requires_grad = requires_grad
    model.fc_original.weight.requires_grad = False  # original remains frozen


def compute_generation_sdm_for_eval(options, model, main_device, gen_ai_model, tokenizer, eval_file, llmType,
                                    generation_output_label="",
                                    load_final_llm_weights=True):
    print(f"Evaluating the held-out set as acceptance/rejection.")
    assert len(model.trueClass_To_dCDF) > 0
    meta_data, _ = \
        utils_gen.get_metadata_lines_for_gen(
            options, gen_ai_model, tokenizer, options.max_length, eval_file,
            calculate_summary_stats=False, is_training=True,
            taskCategory=None,
            modelCategory=utils_gen.modelCategories.classification_with_generation__document_level,
            top_logits_k=None,
            model=model,
            return_text=True,
            llmType=llmType,
            eval_label="Eval (generation decoded)",
            load_final_llm_weights=load_final_llm_weights
        )
    # for generated_message in generated_lines:
    #     print(generated_message)
    eval_classification_embeddings = meta_data["embeddings"].to(main_device)
    # eval_verification_labels = torch.tensor(meta_data["labels"]).to(main_device)  # See note below, as a reminder.
    # # START retrieve global uncertainty stats
    global_uncertainty_statistics = utils_model.load_global_uncertainty_statistics_from_disk(options.model_dir)
    min_valid_qbin_for_class_conditional_accuracy_with_bounded_error = \
        global_uncertainty_statistics.get_min_valid_qbin_with_bounded_error(
            model.min_valid_qbin_for_class_conditional_accuracy)
    predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin = \
        global_uncertainty_statistics.get_summarized_output_magnitude_structure_with_bounded_error_lower_offset_by_bin()
    # # END global uncertainty stats
    acceptance_and_rejection_by_predicted_class = {}
    acceptance_and_rejection_by_predicted_class__and_valid_class_conditional = {}
    for label in range(model.numberOfClasses):
        acceptance_and_rejection_by_predicted_class[label] = 0
        acceptance_and_rejection_by_predicted_class__and_valid_class_conditional[label] = 0

    count_accepted_among_valid_index_conditional_and_task_true = 0
    reliability_indicators = []
    probability_of_acceptances = []
    correct_task_predictions = []
    underlying_task_accuracy_among_index_conditional_by_class_conditional_true_label = {}
    underlying_task_accuracy_among_index_conditional_by_prediction_conditional_predicted_label = {}
    for class_i in range(model.numberOfClasses):
        underlying_task_accuracy_among_index_conditional_by_class_conditional_true_label[class_i] = []
        underlying_task_accuracy_among_index_conditional_by_prediction_conditional_predicted_label[class_i] = []
    index_conditional_instances_with_parsing_errors = 0
    line_id = 0
    for eval_classification_embedding in eval_classification_embeddings:
        # Remember eval verification labels are distinct from the labels of the underlying task.
        # In this case where we are generating
        # from the prefix, the model could correctly answer a prompt assigned for negative construction, so
        # exact match with 'label' will not necessarily be meaningful. If we have
        # the true underlying labels for the tasks represented in the instruction fine-tuning set (e.g.,
        # if the instructions are classification, or code that can be evaluated, etc.) we can use those labels in
        # addition to the index-conditional verification estimate. In
        # this case, we consider the case where we will seek to maximize (as the criteria for early
        # stopping) the count of valid
        # index-conditional predictions of the accepted verification label (i.e., 1) for which the true
        # task prediction is correct (serving as a
        # 'hard reward' from the underlying binary tasks).
        # (For reference, utils_gen.get_metadata_lines_for_gen()
        # also calculates the per-class accuracy of the underlying tasks.)
        prediction_meta_data = \
            model(eval_classification_embedding.unsqueeze(0),
                  forward_type=constants.FORWARD_TYPE_SINGLE_PASS_TEST,
                  min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=
                  min_valid_qbin_for_class_conditional_accuracy_with_bounded_error,
                  predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=
                  predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin)
        prediction_conditional_distribution = \
            prediction_meta_data["rescaled_prediction_conditional_distribution__lower"]
        predicted_class = prediction_meta_data["prediction"]  # 0: not verified; 1: verified (i.e., not the underlying classification task)
        acceptance_and_rejection_by_predicted_class[predicted_class] += 1

        # task prediction is parsed from the text:
        task_prediction = meta_data["task_predictions"][line_id]  # prediction for underlying task by parsing the JSON output
        true_task_label = meta_data["original_labels"][line_id]  # underlying task label
        correct_task_predictions.append(task_prediction == true_task_label)

        if prediction_meta_data["is_valid_index_conditional__lower"]:
            acceptance_and_rejection_by_predicted_class__and_valid_class_conditional[predicted_class] += 1
            if predicted_class == 1:  # verification prediction is 1
                if data_validator.isKnownValidLabel(label=task_prediction, numberOfClasses=model.numberOfClasses):
                    underlying_task_accuracy_among_index_conditional_by_class_conditional_true_label[
                        true_task_label].append(
                        task_prediction == true_task_label
                    )
                    underlying_task_accuracy_among_index_conditional_by_prediction_conditional_predicted_label[
                        task_prediction].append(
                        task_prediction == true_task_label
                    )
                    if task_prediction == true_task_label:
                        count_accepted_among_valid_index_conditional_and_task_true += 1
                else:
                    index_conditional_instances_with_parsing_errors += 1

        reliability_indicators.append(prediction_meta_data["is_valid_index_conditional__lower"])

        prediction_conditional_estimate_of_predicted_class = prediction_conditional_distribution[predicted_class].item()
        if predicted_class == 1:
            probability_of_acceptances.append(prediction_conditional_estimate_of_predicted_class)
        else:
            probability_of_acceptances.append(0.0)  # hard threshold
        line_id += 1

    total_n = \
        sum([acceptance_and_rejection_by_predicted_class[x] for x in
             acceptance_and_rejection_by_predicted_class.keys()])
    print(f"\tEval (generation decoded): Rejected: {acceptance_and_rejection_by_predicted_class[0]} out of "
          f"{total_n}")
    print(f"\tEval (generation decoded): Accepted: {acceptance_and_rejection_by_predicted_class[1]} out of "
          f"{total_n}")
    total_valid_class_conditional_n = \
        sum([acceptance_and_rejection_by_predicted_class__and_valid_class_conditional[x] for x in
             acceptance_and_rejection_by_predicted_class__and_valid_class_conditional.keys()])
    print(f"\tEval (generation decoded): Rejected (among valid index-conditional): "
          f"{acceptance_and_rejection_by_predicted_class__and_valid_class_conditional[0]} out of "
          f"{total_valid_class_conditional_n}")
    count_accepted_among_valid_class_conditional = \
        acceptance_and_rejection_by_predicted_class__and_valid_class_conditional[1]
    print(f"\tEval (generation decoded): Accepted (among valid index-conditional): "
          f"{count_accepted_among_valid_class_conditional} out of "
          f"{total_valid_class_conditional_n}")
    print(f"\tSTOPPING CRITERIA: Eval (generation decoded): Accepted (among valid index-conditional) and CORRECT "
          f"task prediction: "
          f"{count_accepted_among_valid_index_conditional_and_task_true} out of "
          f"{count_accepted_among_valid_class_conditional}")

    print(f"Index-conditional instances with parsing errors: {index_conditional_instances_with_parsing_errors}")

    print_summary(f"Marginal accuracy, across underlying tasks (i.e., not verification):",
                  correct_task_predictions,
                  total=None)
    for class_i in range(model.numberOfClasses):
        print_summary(f"Among index-conditional verification for which verification prediction is 1, "
                      f"Class-conditional task label {class_i} accuracy:",
                      underlying_task_accuracy_among_index_conditional_by_class_conditional_true_label[class_i],
                      total=len(correct_task_predictions))
    for class_i in range(model.numberOfClasses):
        print_summary(f"Among index-conditional verification for which verification prediction is 1, "
                      f"Prediction-conditional task label {class_i} accuracy:",
                      underlying_task_accuracy_among_index_conditional_by_prediction_conditional_predicted_label[class_i],
                      total=len(correct_task_predictions))

    utils_model.save_generated_lines(options.generation_directory, eval_file,
                                     meta_data["generated_lines"], meta_data["uuids"],
                                     meta_data["taskCategoryInts"],
                                     reliability_indicators, probability_of_acceptances,
                                     generation_output_label,
                                     meta_data["task_predictions"],
                                     meta_data["original_labels"],
                                     correct_task_predictions)

    return count_accepted_among_valid_index_conditional_and_task_true


def compute_generation_sdm(options, model, train_token_level_embeddings, train_token_level_uuids,
                           train_shifted_token_labels, main_device, gen_ai_model, tokenizer, train_file,
                           llmType, train_classification_embeddings=None, document_level_uuid2idx=None,
                           load_final_llm_weights=True):
    print(f"Computing Similarity and Distance")
    assert len(model.train_trueClass_To_dCDF) > 0
    if train_classification_embeddings is None:
        meta_data, _ = \
            utils_gen.get_metadata_lines_for_gen(
                options, gen_ai_model, tokenizer, options.max_length, train_file,
                calculate_summary_stats=False, is_training=True,
                taskCategory=None,
                modelCategory=utils_gen.modelCategories.classification_with_generation__document_level,
                top_logits_k=None,
                model=model,
                return_text=True,
                llmType=llmType,
                eval_label="TRAIN (generation decoded)",
                load_final_llm_weights=load_final_llm_weights
            )
        document_level_uuid2idx = meta_data["uuid2idx"]
        train_classification_embeddings = meta_data["embeddings"].to(main_device)
    # Here, we classify the dense representation of the generated output. In contrast, the support set is
    # force-decoded against the available data (both valid examples and constructed negatives).
    _, train_batch_f_positive_outputs, _, \
        train_exemplar_vectors = \
        utils_classification.global_eval(options, model, eval_embeddings=train_classification_embeddings,
                                         eval_labels=None,
                                         split_label="TRAIN (generation decoded)", return_exemplar_vectors=True,
                                         dataset_q=None,
                                         dataset_distance_quantile_per_class=None, main_device=main_device
                                         )  # eval not needed on this pass; hence, labels are None
    # Although matching with training, in this case, the match does not (necessarily) include self,
    # since we are matching the newly generated output against the existing force decoded output.
    # In this case, an exact match is preferable, since
    # achieving that match requires generating the desired output. (Contrast this setting with the classification
    # setting where we explicitly drop the self match since the output remains fixed.)
    train_top_k_distances, train_top_k_distances_idx = \
        model.get_top_support_distances(train_exemplar_vectors.detach().numpy())
    # is_training_support=False because (as noted above), the identity match is not applicable here.
    # Get training distance quantiles, using distance empirical CDF over *force-decoded training*.
    doc_level_train_dataset_q_values, _, doc_level_train_dataset_distance_quantile_per_class = \
        model.get_summary_stats_for_eval(train_batch_f_positive_outputs.shape[0],
                                         train_top_k_distances,
                                         train_top_k_distances_idx,
                                         train_batch_f_positive_outputs,
                                         is_training_support=False,
                                         train_trueClass_To_dCDF=model.train_trueClass_To_dCDF)
    train_dataset_distance_quantile_per_class = []
    train_dataset_q_values = []
    assert train_token_level_embeddings.shape[0] == len(train_token_level_uuids)
    acceptance_by_class_among_non_negative = {}
    for label in range(model.numberOfClasses):
        acceptance_by_class_among_non_negative[label] = set()  # document-level uuids
    doc_level_train_predicted_labels = torch.argmax(train_batch_f_positive_outputs, dim=1)
    doc_id_not_found_count = 0
    for token_index in range(train_token_level_embeddings.shape[0]):
        existing_distribution_indicator = int(train_shifted_token_labels[token_index].item() >= model.gen_ai_vocab)
        # print(f"verification label: {verification_label}")
        # If there are existing_distribution_indicator == 0 examples (for example, over prefixes, prompts,
        # or existing good
        # generations), we will use
        # standard softmax over the original model for regularization:
        if existing_distribution_indicator == 0:  # optional case, see note above
            train_dataset_distance_quantile_per_class.append(torch.ones(1, model.numberOfClasses))
            train_dataset_q_values.append(torch.zeros(1, 1) + (np.e - model.q_rescale_offset))
        else:  # otherwise, we determine sdm based on the prediction
            doc_id_for_token: str = train_token_level_uuids[token_index]
            if doc_id_for_token not in document_level_uuid2idx:
                train_dataset_distance_quantile_per_class.append(torch.ones(1, model.numberOfClasses))
                train_dataset_q_values.append(torch.zeros(1, 1) + (np.e - model.q_rescale_offset))
                doc_id_not_found_count += 1
                continue
            index_into_doc_level_structures: int = document_level_uuid2idx[doc_id_for_token]
            train_predicted_label = doc_level_train_predicted_labels[index_into_doc_level_structures].item()
            """
            if False:  # train_predicted_label == 0:
                # In some applications, it may make sense to further penalize some instances,
                # but we do not do that here.
                train_dataset_distance_quantile_per_class.append(torch.ones(1, model.numberOfClasses))
                train_dataset_q_values.append(torch.zeros(1, 1) + (np.e - model.q_rescale_offset))
            else:
            """
            train_dataset_distance_quantile_per_class.append(
                doc_level_train_dataset_distance_quantile_per_class[index_into_doc_level_structures].unsqueeze(0))
            train_dataset_q_values.append(doc_level_train_dataset_q_values[index_into_doc_level_structures].unsqueeze(1))
            acceptance_by_class_among_non_negative[train_predicted_label].add(doc_id_for_token)  # Note the set operation, since assess at document level
    print(f"Count of missing document id's: {doc_id_not_found_count}")
    # This is different from the eval case. For training, label 0 is the known prefix data
    # (or otherwise acceptable generations from the existing model). The following counts
    # acceptance/rejection at the *document level* among those for which the underlying verification label is 1.
    # The stratification by
    # label 1 occurs in utils_gen.get_metadata_lines_for_gen( with
    # modelCategory=utils_gen.modelCategories.classification_with_generation__document_level ).
    print(f"Train (generation decoded): Rejected: {len(acceptance_by_class_among_non_negative[0])} out of "
          f"{len(acceptance_by_class_among_non_negative[0]) + len(acceptance_by_class_among_non_negative[1])}")
    print(
        f"Train (generation decoded): Accepted: {len(acceptance_by_class_among_non_negative[1])} out of "
        f"{len(acceptance_by_class_among_non_negative[0]) + len(acceptance_by_class_among_non_negative[1])}")
    train_dataset_distance_quantile_per_class = torch.cat(train_dataset_distance_quantile_per_class, dim=0).to(main_device)
    train_dataset_q_values = torch.cat(train_dataset_q_values, dim=0).to(main_device)
    return train_dataset_distance_quantile_per_class, train_dataset_q_values


def train(options, train_file, calibration_file, gen_ai_model, tokenizer, llmType, train_token_level_embeddings=None,
          train_shifted_token_labels=None,
          train_token_level_uuids=None,
          use_balanced_accuracy=False,
          main_device=None, model_dir=None, model=None,
          apply_l2_distribution_regularization=True
          ):
    print("save optimizer")
    kUSE_STANDARD_CROSS_ENTROPY = False  # only for debugging; typically should be False
    if apply_l2_distribution_regularization:
        pdist = nn.PairwiseDistance(p=2)  # regularizer against reference model; typically should be used

    assert model is not None
    print(f"Placing model on {main_device}")
    model = model.to(main_device)

    train_size = train_token_level_embeddings.shape[0]

    print("Starting training")
    start_time = time.time()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=options.learning_rate, betas=(0.9, 0.999), eps=1e-08)

    criterion = nn.NLLLoss()

    all_epoch_cumulative_losses = []

    batch_size = options.batch_size

    set_genai_gradient_update_state_inplace(model, requires_grad=True)

    use_sdm_as_regularizer = True
    print(f"apply_l2_distribution_regularization: {apply_l2_distribution_regularization}")
    if apply_l2_distribution_regularization:
        assert options.gen_ai_training_min_beta >= 0
        assert options.gen_ai_training_max_beta >= options.gen_ai_training_min_beta
        print(f"Training with a beta schedule in "
              f"[{options.gen_ai_training_min_beta}, {options.gen_ai_training_max_beta}]")
    utils_model.save_llm_weights_for_mlx_generation(options, model, save_as_final_llm_weights=True)
    if use_sdm_as_regularizer:
        train_dataset_distance_quantile_per_class, train_dataset_q_values = \
            compute_generation_sdm(options, model, train_token_level_embeddings, train_token_level_uuids,
                               train_shifted_token_labels, main_device, gen_ai_model, tokenizer, train_file,
                               llmType, train_classification_embeddings=None, document_level_uuid2idx=None,
                                   load_final_llm_weights=True)
        default_training_q_values = (
                    torch.zeros(train_token_level_embeddings.shape[0], 1) + (np.e - model.q_rescale_offset)).to(main_device)
    else:  # equivalent to using standard cross-entropy with softmax
        train_dataset_q_values = \
            (torch.zeros(train_token_level_embeddings.shape[0], 1) + (np.e - model.q_rescale_offset)).to(main_device)
        train_dataset_distance_quantile_per_class = None
        default_training_q_values = \
            (torch.zeros(train_token_level_embeddings.shape[0], 1) + (np.e - model.q_rescale_offset)).to(main_device)

    calibration_count_accepted_among_valid_class_conditional = \
        compute_generation_sdm_for_eval(options, model, main_device, gen_ai_model, tokenizer, calibration_file, llmType,
                                        generation_output_label="init_",
                                        load_final_llm_weights=True)
    max_dev_count_accepted_among_valid_class_conditional = calibration_count_accepted_among_valid_class_conditional
    max_dev_count_accepted_among_valid_class_conditional_epoch = -1
    for e in range(options.epoch):
        # shuffle data
        # TODO: add option for permutation to document-level.
        shuffled_train_indexes = torch.randperm(train_token_level_embeddings.shape[0]).to(main_device)
        shuffled_train_token_level_embeddings = train_token_level_embeddings[shuffled_train_indexes]
        shuffled_train_genai_shifted_augmented_labels = train_shifted_token_labels[shuffled_train_indexes]

        if e < options.warm_up_epochs:
            print(f"Epoch {e} is set to calculate standard softmax.")
            shuffled_q = default_training_q_values[shuffled_train_indexes].to(main_device)
            shuffled_distance_quantile_per_class = None
        else:
            shuffled_q = train_dataset_q_values[shuffled_train_indexes].to(main_device)
            if train_dataset_distance_quantile_per_class is None:
                shuffled_distance_quantile_per_class = None
            else:
                shuffled_distance_quantile_per_class = \
                    train_dataset_distance_quantile_per_class[shuffled_train_indexes].to(main_device)

        batch_num = 0
        cumulative_losses = []

        total_mini_batches = len(range(0, train_size, batch_size))
        beta = options.gen_ai_training_min_beta
        beta_step = (options.gen_ai_training_max_beta-options.gen_ai_training_min_beta) / total_mini_batches

        for i in range(0, train_size, batch_size):
            if i % max(1.0, int((train_size * 0.10) / batch_size)) * batch_size == 0:
                print(f"TRAINING LOOP: Currently processing {i} of {train_size}")
            batch_num += 1
            batch_range = min(batch_size, train_size - i)

            batch_x = shuffled_train_token_level_embeddings[i:i + batch_range]
            batch_genai_y = shuffled_train_genai_shifted_augmented_labels[i:i + batch_range]
            batch_q = shuffled_q[i:i + batch_range]
            if shuffled_distance_quantile_per_class is not None:
                batch_distance_quantile_per_class = shuffled_distance_quantile_per_class[i:i + batch_range]
            else:
                batch_distance_quantile_per_class = None
            optimizer.zero_grad()
            model.train()

            batch_f_genai, batch_f_original = model(batch_x, batch_q, batch_f_positive=None,
                                                    batch_distance_quantile_per_class=batch_distance_quantile_per_class,
                                                    forward_type=
                                                    constants.FORWARD_TYPE_GENAI_WITH_ROUTER_TOKEN_LEVEL_PREDICTION,
                                                    train=True)

            if apply_l2_distribution_regularization:
                with torch.no_grad():
                    # We lessen regularization on the peak of the distribution by not considering the most probable
                    # indexes of the negative, positive, and original distributions in the L2 constraint. We also do the
                    # same for the ground-truth index.
                    top_events_k = 1
                    top_k_sort_by_largest = True
                    neg_original_max_half_distribution_i = torch.topk(batch_f_original[:, 0:model.gen_ai_vocab],
                                                                      top_events_k, dim=1, largest=top_k_sort_by_largest)[1]
                    pos_original_max_half_distribution_i = torch.topk(batch_f_original[:, -model.gen_ai_vocab:],
                                                                      top_events_k, dim=1, largest=top_k_sort_by_largest)[1] + model.gen_ai_vocab  # note the offset
                    negative_max_half_distribution_i = torch.topk(batch_f_genai[:, 0:model.gen_ai_vocab],
                                                                  top_events_k, dim=1, largest=top_k_sort_by_largest)[1]
                    positive_max_half_distribution_i = torch.topk(batch_f_genai[:, -model.gen_ai_vocab:],
                                                                  top_events_k, dim=1, largest=top_k_sort_by_largest)[1] + model.gen_ai_vocab  # note the offset
                    distribution_mass_mask = (
                            torch.ones_like(batch_f_genai).scatter_(1, neg_original_max_half_distribution_i, 0.0) *
                            torch.ones_like(batch_f_genai).scatter_(1, pos_original_max_half_distribution_i, 0.0) *
                            torch.ones_like(batch_f_genai).scatter_(1, negative_max_half_distribution_i, 0.0) *
                            torch.ones_like(batch_f_genai).scatter_(1, positive_max_half_distribution_i, 0.0) *
                            torch.ones_like(batch_f_genai).scatter_(1, batch_genai_y.unsqueeze(1), 0.0)
                    ).to(batch_f_genai.device)

            if apply_l2_distribution_regularization:
                regularization_term = pdist(
                    distribution_mass_mask * batch_f_original,
                    distribution_mass_mask * batch_f_genai).mean()

                llm_loss = criterion(batch_f_genai, batch_genai_y)

                with torch.no_grad():  # rescaling factor for the regularization term
                    regularization_scale_term = (torch.log(llm_loss + model.kEPS) /
                                                 (torch.log(regularization_term + model.kEPS) + model.kEPS)
                                                 ).item()
                loss = llm_loss + beta * torch.sqrt(
                    torch.clamp(regularization_term, min=1.0) ** min(max(regularization_scale_term, 0.0), 1.0))
            else:
                loss = criterion(batch_f_genai, batch_genai_y)

            assert not torch.logical_or(torch.isnan(loss), torch.isinf(loss)).item()
            cumulative_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            beta += beta_step
        print(f"---------------Epoch: {e + 1}---------------")
        print(f"Epoch average loss: {np.mean(cumulative_losses)}")
        all_epoch_cumulative_losses.extend(cumulative_losses)
        print(f"Average loss across all mini-batches (all epochs): {np.mean(all_epoch_cumulative_losses)}")
        # We retain a distinction between the running LLM weights and the best epoch LLM weights via
        # the `save_as_final_llm_weights` argument --- and the corresponding `load_final_llm_weights` argument of
        # `compute_generation_sdm` and `compute_generation_sdm_for_eval`.
        utils_model.save_llm_weights_for_mlx_generation(options, model, save_as_final_llm_weights=False)  # run before mlx generation

        if use_sdm_as_regularizer:
            train_dataset_distance_quantile_per_class, train_dataset_q_values = \
                compute_generation_sdm(options, model, train_token_level_embeddings, train_token_level_uuids,
                                       train_shifted_token_labels, main_device, gen_ai_model, tokenizer, train_file,
                                       llmType, train_classification_embeddings=None, document_level_uuid2idx=None,
                                       load_final_llm_weights=False)

        calibration_count_accepted_among_valid_class_conditional = \
            compute_generation_sdm_for_eval(options, model, main_device, gen_ai_model, tokenizer, calibration_file,
                                            llmType,
                                            generation_output_label=f"epoch_{e + 1}",
                                            load_final_llm_weights=False)

        if calibration_count_accepted_among_valid_class_conditional > max_dev_count_accepted_among_valid_class_conditional:
            max_dev_count_accepted_among_valid_class_conditional = calibration_count_accepted_among_valid_class_conditional
            max_dev_count_accepted_among_valid_class_conditional_epoch = e + 1
            is_best_running_epoch = True
        else:
            is_best_running_epoch = False

        if is_best_running_epoch:
            print(">>Saving as current best epoch<<")
            # model.increment_model_calibration_training_stage(set_value=1)  # TODO: add LLM training to schedule
            utils_model.save_model(model, model_dir)
            logger.info(f"Model saved to {model_dir}")
            utils_model.save_llm_weights_for_mlx_generation(options, model, save_as_final_llm_weights=True)

        print(f"Epoch: {e + 1} / Calibration set count of reliable class-conditional estimates of acceptance: "
              f"{calibration_count_accepted_among_valid_class_conditional}")

        print(f"\tCurrent max Calibration set count of reliable class-conditional estimates of acceptance: "
              f"{max_dev_count_accepted_among_valid_class_conditional} at epoch "
              f"{max_dev_count_accepted_among_valid_class_conditional_epoch}")

    print(f"\tFINAL Max Calibration set count of reliable class-conditional estimates of acceptance: "
              f"{max_dev_count_accepted_among_valid_class_conditional} at epoch "
              f"{max_dev_count_accepted_among_valid_class_conditional_epoch}")

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")



