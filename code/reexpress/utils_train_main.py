# Copyright Reexpress AI, Inc. All rights reserved.

from sdm_model import SimilarityDistanceMagnitudeCalibrator
import constants
import utils_model
import utils_pretraining_initialization
import utils_calibrate
import utils_eval_batch

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

import logging
import sys
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def print_timing_to_standard_out(label, elapsed_time, print_timing=False):
    if print_timing and elapsed_time is not None:
        print(f"[TIMING] (seconds): {label}: {time.time() - elapsed_time}")


def train(options, train_embeddings=None, calibration_embeddings=None,
          train_labels=None, calibration_labels=None,
          model_params=None,
          main_device=None, model_dir=None, model=None, shuffle_index=0):
    print(f"Training on main device: {main_device}")
    current_device = main_device

    if model is None:
        print("Initializing model")
        model = SimilarityDistanceMagnitudeCalibrator(**model_params).to(current_device)
    else:
        model = model.to(current_device)
    if options.pretraining_initialization_epochs > 0:
        # The train and calibration data for the iterative shuffle and SDM loss
        # (i.e., train_embeddings and calibration_embeddings) are assumed to be disjoint from
        # options.pretraining_initialization_tensors_file
        held_out_embeddings = torch.cat([train_embeddings, calibration_embeddings], dim=0)  # 0 is batch dim
        held_out_labels = torch.cat([train_labels, calibration_labels], dim=0)
        model = utils_pretraining_initialization.pretrain(options, model=model, model_dir=model_dir,
                                                          held_out_embeddings=held_out_embeddings,
                                                          held_out_labels=held_out_labels,
                                                          ).to(current_device)
        del held_out_embeddings
        del held_out_labels

    if options.ood_support_file.strip() != "":
        import utils_preprocess
        import data_validator
        ood_support_meta_data, _ = \
            utils_preprocess.get_metadata_lines(options, options.ood_support_file,
                                                reduce=False,
                                                use_embeddings=options.use_embeddings,
                                                concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                calculate_summary_stats=False, is_training=False)
        ood_support_embeddings = ood_support_meta_data["embeddings"]
        ood_support_labels = ood_support_meta_data["labels"]
        label_parity_warning = False
        for ood_label in ood_support_labels:
            if ood_label != data_validator.oodLabel:
                label_parity_warning = True
        if label_parity_warning:
            print(f">>NOTE: Using --ood_support_file is primarily intended for adding OOD "
                  f"(label=={data_validator.oodLabel}) instances to the training database. "
                  f"You can add other instances to the support (as you are doing) using this "
                  f"mechanism, but note that they will not participate in the iterative shuffling with the "
                  f"calibration set. As such, typically documents with labels in [0, C) should instead be added to "
                  f"--input_training_set_file or --input_calibration_set_file.<<")
        ood_support_document_ids = ood_support_meta_data["uuids"]
        print(f"Loaded {len(ood_support_labels)} OOD/additional documents to add to the training support set")
        ood_support_labels = torch.tensor(ood_support_labels)
    else:
        ood_support_meta_data = None

    train_size = train_embeddings.shape[0]

    print("Starting training")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=options.learning_rate, betas=(0.9, 0.999), eps=1e-08)

    criterion = nn.NLLLoss()
    max_dev_acc = 0
    max_dev_acc_epoch = -1
    train_acc_for_max_dev_acc = 0

    max_dev_balanced_acc = 0
    max_dev_balanced_acc_epoch = -1
    train_balanced_acc_for_max_dev_acc = 0

    max_dev_balanced_q = 0
    max_dev_balanced_q_epoch = -1
    train_balanced_q_for_max_dev_balanced_q = 0

    min_dev_balanced_sdm_loss = np.inf
    min_dev_balanced_sdm_loss_epoch = -1
    train_balanced_sdm_loss_for_min_dev_sdm_loss = np.inf

    all_epoch_cumulative_losses = []

    batch_size = options.batch_size

    train_dataset_q_values = torch.zeros(train_embeddings.shape[0], 1) + (np.e - model.q_rescale_offset)
    train_dataset_distance_quantile_per_class = None

    for e in range(options.epoch):
        # shuffle data
        shuffled_train_indexes = torch.randperm(train_embeddings.shape[0])
        shuffled_train_embeddings = train_embeddings[shuffled_train_indexes]
        shuffled_train_labels = train_labels[shuffled_train_indexes]
        shuffled_q = train_dataset_q_values[shuffled_train_indexes]
        if e == 0:  # The initial epoch uses softmax (q=e-2, d=1).
            shuffled_distance_quantile_per_class = None
        else:
            shuffled_distance_quantile_per_class = train_dataset_distance_quantile_per_class[shuffled_train_indexes]
        batch_num = 0
        cumulative_losses = []

        single_epoch_time = time.time() if options.print_timing else None
        for i in range(0, train_size, batch_size):
            batch_num += 1
            batch_range = min(batch_size, train_size - i)

            batch_x = shuffled_train_embeddings[i:i + batch_range].to(current_device)
            batch_y = shuffled_train_labels[i:i + batch_range].to(current_device)
            batch_q = shuffled_q[i:i + batch_range].to(current_device)
            if shuffled_distance_quantile_per_class is not None:
                batch_distance_quantile_per_class = \
                    shuffled_distance_quantile_per_class[i:i + batch_range].to(current_device)
            else:
                batch_distance_quantile_per_class = None
            optimizer.zero_grad()
            model.train()
            _, batch_log_sdm_output = model(batch_x, batch_q,
                                            batch_distance_quantile_per_class=batch_distance_quantile_per_class,
                                            forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION,
                                            train=True)
            if len(batch_log_sdm_output.shape) == 1:
                loss = criterion(batch_log_sdm_output.unsqueeze(0), batch_y)
            else:
                loss = criterion(batch_log_sdm_output, batch_y)

            cumulative_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print_timing_to_standard_out("Single epoch", single_epoch_time, print_timing=options.print_timing)
        print(f"---------------Shuffle Index {shuffle_index}: Epoch: {e + 1}---------------")
        print(f"Epoch average (marginal) cumulative loss: {np.mean(cumulative_losses)}")
        all_epoch_cumulative_losses.extend(cumulative_losses)
        print(f"Average (marginal) loss across all mini-batches (all epochs): {np.mean(all_epoch_cumulative_losses)}")
        support_stats_time = time.time() if options.print_timing else None
        # First get training set predictions and exemplar vectors.
        train_batch_f_positive_outputs, train_exemplar_vectors = \
            utils_eval_batch.get_predictions_and_exemplar_vectors(options.eval_batch_size, model,
                                                                  train_embeddings,
                                                                  current_device,
                                                                  place_output_on_cpu=True)
        # MUST set support set predictions before calculating q, d0:
        model.set_train_predicted_labels(torch.argmax(train_batch_f_positive_outputs, dim=1))

        calibration_batch_f_positive_outputs, calibration_exemplar_vectors = \
            utils_eval_batch.get_predictions_and_exemplar_vectors(options.eval_batch_size, model,
                                                                  calibration_embeddings,
                                                                  current_device,
                                                                  place_output_on_cpu=True)
        model.set_calibration_predicted_labels(torch.argmax(calibration_batch_f_positive_outputs, dim=1))
        if ood_support_meta_data is not None:
            ood_support_batch_f_positive_outputs, ood_support_exemplar_vectors = \
                utils_eval_batch.get_predictions_and_exemplar_vectors(
                    options.eval_batch_size, model,
                    ood_support_embeddings,
                    current_device, place_output_on_cpu=True)

            ood_support_predicted_labels = torch.argmax(ood_support_batch_f_positive_outputs, dim=1)
            _, calibration_top_k_distances, calibration_top_k_distances_idx = \
                model.construct_support_index(support_exemplar_vectors_numpy=train_exemplar_vectors.numpy(),
                                              calibration_exemplar_vectors_numpy=calibration_exemplar_vectors.numpy(),
                                              ood_support_exemplar_vectors_numpy=ood_support_exemplar_vectors.numpy(),
                                              ood_support_labels=ood_support_labels,  # tensor
                                              ood_support_predicted_labels=ood_support_predicted_labels,  # tensor
                                              ood_support_document_ids=ood_support_document_ids
                                              )
        else:
            # Set the exemplar vectors of training as the support set and fetch the calibration distances
            _, calibration_top_k_distances, calibration_top_k_distances_idx = \
                model.construct_support_index(support_exemplar_vectors_numpy=train_exemplar_vectors.numpy(),
                                              calibration_exemplar_vectors_numpy=calibration_exemplar_vectors.numpy())

        # Fetch the training distances. This will include the identity match, which is handled below.
        # Currently, we assume there are no duplicates in the data splits (or at least there are very few)
        train_top_k_distances__including_self, train_top_k_distances_idx__including_self = \
            model.get_top_support_distances(train_exemplar_vectors.numpy())

        # get q values and dCDF for training; is_training_support=True will discard the first (identity) match
        # Note that the distance quantiles for training are determined by distances over training. The class
        # attribute model.trueClass_To_dCDF is over calibration, which is what should be used for new, unseen
        # test instances.
        train_dataset_q_values, train_trueClass_To_dataset_total_q_ood, train_trueClass_To_total_labels, \
            train_dataset_d0_values, train_trueClass_To_dCDF = model.set_summary_stats_for_support_vectorized(
            train_exemplar_vectors.shape[0],
            train_top_k_distances__including_self, train_top_k_distances_idx__including_self,
            train_batch_f_positive_outputs,
            train_labels, is_training_support=True)

        model.set_train_trueClass_To_dCDF(train_trueClass_To_dCDF)

        for class_i in range(model.numberOfClasses):
            if len(train_trueClass_To_dCDF[class_i]) > 0:
                print(f"\tDistances: {'Train'}: (class {class_i}) mean d0: {np.mean(train_trueClass_To_dCDF[class_i])}; "
                      f"median d0: {np.median(train_trueClass_To_dCDF[class_i])}, "
                      f"min: {np.min(train_trueClass_To_dCDF[class_i])}, "
                      f"max: {np.max(train_trueClass_To_dCDF[class_i])}, "
                      f"out of {len(train_trueClass_To_dCDF[class_i])}")
            else:
                print(
                    f"\tDistances: {'Train'}: (class {class_i}): WARNING NO DISTANCES AVAILABLE")
        for class_i in range(model.numberOfClasses):
            print(f"\tTotal OOD q values (q<={model.ood_limit}): {'Train'}: (class {class_i}): "
                  f"{train_trueClass_To_dataset_total_q_ood[class_i]} "
                  f"out of {train_trueClass_To_total_labels[class_i]}: "
                  f"{train_trueClass_To_dataset_total_q_ood[class_i]/(float(train_trueClass_To_total_labels[class_i]) if train_trueClass_To_total_labels[class_i] > 0 else 1.0)}")
        # get q values for calibration and set the class dCDF
        calibration_dataset_q_values, calibration_trueClass_To_dataset_total_q_ood, \
            calibration_trueClass_To_total_labels, calibration_dataset_d0_values, _ = \
            model.set_summary_stats_for_support_vectorized(calibration_exemplar_vectors.shape[0],
                                                           calibration_top_k_distances,
                                                           calibration_top_k_distances_idx,
                                                           calibration_batch_f_positive_outputs,
                                                           calibration_labels,
                                                           is_training_support=False)
        for class_i in range(model.numberOfClasses):
            if len(model.trueClass_To_dCDF[class_i]) > 0:
                print(f"\tDistances: {constants.SPLIT_LABEL_calibration_during_training}: (class {class_i}) mean d0: "
                      f"{np.mean(model.trueClass_To_dCDF[class_i])}; "
                      f"median d0: {np.median(model.trueClass_To_dCDF[class_i])}, "
                      f"min: {np.min(model.trueClass_To_dCDF[class_i])}, "
                      f"max: {np.max(model.trueClass_To_dCDF[class_i])}, "
                      f"out of {len(model.trueClass_To_dCDF[class_i])}")
            else:
                print(
                    f"\tDistances: {constants.SPLIT_LABEL_calibration_during_training}: (class {class_i}): "
                    f"WARNING NO DISTANCES AVAILABLE")
        for class_i in range(model.numberOfClasses):
            print(f"\tTotal OOD q values (q<={model.ood_limit}): {constants.SPLIT_LABEL_calibration_during_training}: (class {class_i}): "
                  f"{calibration_trueClass_To_dataset_total_q_ood[class_i]} "
                  f"out of {calibration_trueClass_To_total_labels[class_i]}: "
                  f"{calibration_trueClass_To_dataset_total_q_ood[class_i]/(float(calibration_trueClass_To_total_labels[class_i]) if calibration_trueClass_To_total_labels[class_i] > 0 else 1.0)}")
        print_timing_to_standard_out("Construct exemplars, q, distances", support_stats_time,
                                     print_timing=options.print_timing)
        train_d_time = time.time() if options.print_timing else None
        # get training distance quantiles, using distance empirical CDF over training
        train_dataset_distance_quantile_per_class = \
            model.get_distance_quantiles_vectorized(train_dataset_d0_values,
                                                    train_trueClass_To_dCDF=train_trueClass_To_dCDF)
        print_timing_to_standard_out("Calculate training distance quantiles", train_d_time,
                                     print_timing=options.print_timing)
        calibration_d_time = time.time() if options.print_timing else None
        # get calibration training quantiles, using distance empirical CDF over calibration
        calibration_dataset_distance_quantile_per_class = \
            model.get_distance_quantiles_vectorized(calibration_dataset_d0_values,
                                                    train_trueClass_To_dCDF=None)

        # utils_unit_tests.run_unit_test_comparison_of_get_distance_quantiles(model, calibration_dataset_d0_values,
        #                                                    train_dataset_d0_values, train_trueClass_To_dCDF)
        print_timing_to_standard_out("Calculate calibration distance quantiles", calibration_d_time,
                                     print_timing=options.print_timing)
        eval_time = time.time() if options.print_timing else None
        # Calculate metrics from the cached output (the CNN is not rerun).
        # In the current version, we do not reset train_dataset_q_values for predictions flips resulting after
        # rescaling, since they are very rare and will be handled otherwise by low q and d. However, as with earlier
        # versions, the prediction is always determined by f rather than the sdm output.
        train_per_class_loss_as_list, train_balanced_loss, train_marginal_loss, \
            train_per_class_accuracy_as_list, train_balanced_accuracy, train_marginal_accuracy, \
            train_per_class_q_as_list, train_balanced_q, train_marginal_q, \
            train_sdm_outputs = \
            utils_eval_batch.get_metrics_from_cached_outputs(options.eval_batch_size, model,
                                                             train_batch_f_positive_outputs,
                                                             current_device,
                                                             train_labels,
                                                             q_values=train_dataset_q_values,
                                                             distance_quantile_per_class=train_dataset_distance_quantile_per_class)

        calibration_per_class_loss_as_list, calibration_balanced_loss, calibration_marginal_loss, \
            calibration_per_class_accuracy_as_list, calibration_balanced_accuracy, calibration_marginal_accuracy, \
            calibration_per_class_q_as_list, calibration_balanced_q, calibration_marginal_q, \
            calibration_sdm_outputs = \
            utils_eval_batch.get_metrics_from_cached_outputs(options.eval_batch_size, model,
                                                             calibration_batch_f_positive_outputs,
                                                             current_device,
                                                             calibration_labels,
                                                             q_values=calibration_dataset_q_values,
                                                             distance_quantile_per_class=calibration_dataset_distance_quantile_per_class)
        print_timing_to_standard_out("Calculate metrics", eval_time,
                                     print_timing=options.print_timing)
        time_to_set_similarity = time.time() if options.print_timing else None
        utils_calibrate.set_model_rescaled_similarity_vectorized(model, calibration_batch_f_positive_outputs,
                                                                 calibration_dataset_q_values, calibration_sdm_outputs)

        print_timing_to_standard_out("Set model rescaled Similarity", time_to_set_similarity,
                                     print_timing=options.print_timing)
        utils_eval_batch.print_metrics(e=e, numberOfClasses=model.numberOfClasses, split_label_name="Training set",
                                       per_class_loss_as_list=train_per_class_loss_as_list,
                                       balanced_loss=train_balanced_loss,
                                       marginal_loss=train_marginal_loss,
                                       per_class_accuracy_as_list=train_per_class_accuracy_as_list,
                                       balanced_accuracy=train_balanced_accuracy,
                                       marginal_accuracy=train_marginal_accuracy,
                                       per_class_q_as_list=train_per_class_q_as_list,
                                       balanced_q=train_balanced_q,
                                       marginal_q=train_marginal_q)

        utils_eval_batch.print_metrics(e=e, numberOfClasses=model.numberOfClasses, split_label_name="Calibration set",
                                       per_class_loss_as_list=calibration_per_class_loss_as_list,
                                       balanced_loss=calibration_balanced_loss,
                                       marginal_loss=calibration_marginal_loss,
                                       per_class_accuracy_as_list=calibration_per_class_accuracy_as_list,
                                       balanced_accuracy=calibration_balanced_accuracy,
                                       marginal_accuracy=calibration_marginal_accuracy,
                                       per_class_q_as_list=calibration_per_class_q_as_list,
                                       balanced_q=calibration_balanced_q,
                                       marginal_q=calibration_marginal_q)

        is_best_running_epoch = calibration_balanced_loss <= min_dev_balanced_sdm_loss

        if calibration_balanced_loss <= min_dev_balanced_sdm_loss:
            min_dev_balanced_sdm_loss = calibration_balanced_loss
            min_dev_balanced_sdm_loss_epoch = e + 1
            train_balanced_sdm_loss_for_min_dev_sdm_loss = train_balanced_loss

        if calibration_marginal_accuracy >= max_dev_acc:
            max_dev_acc = calibration_marginal_accuracy
            max_dev_acc_epoch = e + 1
            train_acc_for_max_dev_acc = train_marginal_accuracy

        if calibration_balanced_accuracy >= max_dev_balanced_acc:
            max_dev_balanced_acc = calibration_balanced_accuracy
            max_dev_balanced_acc_epoch = e + 1
            train_balanced_acc_for_max_dev_acc = train_balanced_accuracy

        if calibration_balanced_q >= max_dev_balanced_q:
            max_dev_balanced_q = calibration_balanced_q
            max_dev_balanced_q_epoch = e + 1
            train_balanced_q_for_max_dev_balanced_q = train_balanced_q

        if is_best_running_epoch:
            model.increment_model_calibration_training_stage(set_value=1)
            utils_model.save_model(model, model_dir)
            logger.info(f"Model saved at {model_dir} as best running epoch.")
        print(f"---Summary---")
        print(f"\tCurrent max Calibration set accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch} "
              f"(corresponding Training set accuracy: {train_acc_for_max_dev_acc})")
        print(f"\tCurrent max Calibration set Balanced accuracy: {max_dev_balanced_acc} at epoch {max_dev_balanced_acc_epoch} "
              f"(corresponding Training set Balanced accuracy: {train_balanced_acc_for_max_dev_acc})")
        print(f"\tCurrent max Calibration set Balanced q: {max_dev_balanced_q} at epoch {max_dev_balanced_q_epoch} "
              f"(corresponding Training set Balanced q: {train_balanced_q_for_max_dev_balanced_q})")
        print(f"\tCurrent min Calibration set Balanced SDM loss: {min_dev_balanced_sdm_loss} at epoch {min_dev_balanced_sdm_loss_epoch} "
              f"(corresponding Training set Balanced SDM loss: {train_balanced_sdm_loss_for_min_dev_sdm_loss})")

    print(f"+++++++++++++++Shuffle Index {shuffle_index}: Training complete+++++++++++++++")
    print(f"\tMax Calibration set accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch} "
          f"(corresponding Training set accuracy: {train_acc_for_max_dev_acc})")
    print(
        f"\tMax Calibration set Balanced accuracy: {max_dev_balanced_acc} at epoch {max_dev_balanced_acc_epoch} "
        f"(corresponding Training set Balanced accuracy: {train_balanced_acc_for_max_dev_acc})")
    print(
        f"\tMax Calibration set Balanced q: {max_dev_balanced_q} at epoch {max_dev_balanced_q_epoch} "
        f"(corresponding Training set Balanced q: {train_balanced_q_for_max_dev_balanced_q})")
    print(
        f"\tMin Calibration set Balanced SDM loss: {min_dev_balanced_sdm_loss} at epoch {min_dev_balanced_sdm_loss_epoch} "
        f"(corresponding Training set Balanced SDM loss: {train_balanced_sdm_loss_for_min_dev_sdm_loss})")

    print(f"Final epoch chosen based on the minimum Balanced SDM loss (over calibration).")

    print(f"Reloading best model to calibrate based on the provided alpha value.")
    min_rescaled_similarity_to_determine_high_reliability_region = \
        utils_calibrate.calibrate_to_determine_high_reliability_region(options, model_dir=model_dir)
    return max_dev_balanced_acc, max_dev_balanced_q, min_dev_balanced_sdm_loss, \
        min_rescaled_similarity_to_determine_high_reliability_region
