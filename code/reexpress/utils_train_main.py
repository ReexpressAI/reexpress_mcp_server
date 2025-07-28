# Copyright Reexpress AI, Inc. All rights reserved.

from sdm_model import SimilarityDistanceMagnitudeCalibrator
import constants
import utils_classification
import utils_model
import utils_pretraining_initialization

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


def train(options, train_embeddings=None, calibration_embeddings=None,
          train_labels=None, calibration_labels=None,
          model_params=None, use_balanced_accuracy=False,
          main_device=None, model_dir=None, model=None):

    kUSE_STANDARD_CROSS_ENTROPY = False  # only for debugging; typically should be False

    if model is None:
        print("Initializing model")
        model = SimilarityDistanceMagnitudeCalibrator(**model_params).to(main_device)
    if options.pretraining_initialization_epochs > 0:
        # The train and calibration data for the iterative shuffle and SDM loss
        # (i.e., train_embeddings and calibration_embeddings) are assumed to be disjoint from
        # options.pretraining_initialization_tensors_file
        held_out_embeddings = torch.cat([train_embeddings, calibration_embeddings], dim=0)  # 0 is batch dim
        held_out_labels = torch.cat([train_labels, calibration_labels], dim=0)
        model = utils_pretraining_initialization.pretrain(options, model=model, model_dir=model_dir,
                                                          held_out_embeddings=held_out_embeddings,
                                                          held_out_labels=held_out_labels,
                                                          ).to(main_device)
        del held_out_embeddings
        del held_out_labels
    if model.is_gen_ai:
        print(f"FREEZING LLM DISTRIBUTION WEIGHTS")
        model.fc_negative.weight.requires_grad = False
        model.fc_positive.weight.requires_grad = False
        model.fc_original.weight.requires_grad = False

    if options.ood_support_file.strip() != "":
        import utils_preprocess
        import data_validator
        ood_support_meta_data, _ = \
            utils_preprocess.get_metadata_lines(options, options.ood_support_file,
                                                reduce=False,
                                                use_embeddings=options.use_embeddings,
                                                concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                calculate_summary_stats=False, is_training=False)
        ood_support_embeddings = ood_support_meta_data["embeddings"].to(main_device)
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

    max_dev_balanced_median_q = 0
    max_dev_balanced_median_q_epoch = -1
    train_balanced_median_q_for_max_dev_median_q = 0

    min_dev_balanced_sdm_loss = np.inf
    min_dev_balanced_sdm_loss_epoch = -1
    train_balanced_sdm_loss_for_min_dev_sdm_loss = np.inf

    all_epoch_cumulative_losses = []

    batch_size = options.batch_size
    # num_batch_instances = math.ceil((train_size / options.batch_size))

    train_dataset_q_values = torch.zeros(train_embeddings.shape[0], 1) + (np.e - model.q_rescale_offset)
    train_dataset_distance_quantile_per_class = None
    default_training_q_values = torch.zeros(train_embeddings.shape[0], 1) + (np.e - model.q_rescale_offset)
    for e in range(options.epoch):
        # train_predicted_labels = torch.tensor(model.train_predicted_labels)
        # shuffle data
        shuffled_train_indexes = torch.randperm(train_embeddings.shape[0]).to(main_device)
        shuffled_train_embeddings = train_embeddings[shuffled_train_indexes]
        shuffled_train_labels = train_labels[shuffled_train_indexes]
        if e < options.warm_up_epochs:
            shuffled_q = default_training_q_values[shuffled_train_indexes]
        else:
            shuffled_q = train_dataset_q_values[shuffled_train_indexes]
        if e == 0 or train_dataset_distance_quantile_per_class is None or e < options.warm_up_epochs:
            shuffled_distance_quantile_per_class = None
        else:
            shuffled_distance_quantile_per_class = train_dataset_distance_quantile_per_class[shuffled_train_indexes]
        batch_num = 0
        cumulative_losses = []

        for i in range(0, train_size, batch_size):
            batch_num += 1
            batch_range = min(batch_size, train_size - i)

            batch_x = shuffled_train_embeddings[i:i + batch_range]
            batch_y = shuffled_train_labels[i:i + batch_range]
            batch_q = shuffled_q[i:i + batch_range]
            if shuffled_distance_quantile_per_class is not None:
                batch_distance_quantile_per_class = shuffled_distance_quantile_per_class[i:i + batch_range]
            else:
                batch_distance_quantile_per_class = None
            optimizer.zero_grad()
            model.train()
            _, rescaled_batch_output = model(batch_x, batch_q,
                                             batch_distance_quantile_per_class=batch_distance_quantile_per_class,
                                             forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION,
                                             train=True)
            if len(rescaled_batch_output.shape) == 1:
                loss = criterion(rescaled_batch_output.unsqueeze(0), batch_y)
            else:
                loss = criterion(rescaled_batch_output, batch_y)

            cumulative_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"---------------Epoch: {e + 1}---------------")
        print(f"Epoch average loss: {np.mean(cumulative_losses)}")
        all_epoch_cumulative_losses.extend(cumulative_losses)
        print(f"Average loss across all mini-batches (all epochs): {np.mean(all_epoch_cumulative_losses)}")
        _, train_batch_f_positive_outputs, _, \
            train_exemplar_vectors = \
            utils_classification.global_eval(options, model, eval_embeddings=train_embeddings, eval_labels=None,
                                             split_label="TRAIN", return_exemplar_vectors=True,
                                             dataset_q=None,
                                             dataset_distance_quantile_per_class=None
                                             )  # eval not needed on this pass; hence, labels are None
        # MUST set support set predictions before calculating q, d0:
        model.set_train_predicted_labels(np.argmax(train_batch_f_positive_outputs.detach().numpy(), axis=1))

        _, calibration_batch_f_positive_outputs, _,\
            calibration_exemplar_vectors = \
            utils_classification.global_eval(options, model, eval_embeddings=calibration_embeddings,
                                             eval_labels=None,
                                             split_label=constants.SPLIT_LABEL_calibration_during_training,
                                             return_exemplar_vectors=True,
                                             dataset_q=None,
                                             dataset_distance_quantile_per_class=None
                                             )  # eval not needed on this pass; hence, labels are None
        model.set_calibration_predicted_labels(torch.argmax(calibration_batch_f_positive_outputs, dim=1))
        if ood_support_meta_data is not None:
            _, ood_support_batch_f_positive_outputs, _, \
                ood_support_exemplar_vectors = \
                utils_classification.global_eval(options, model, eval_embeddings=ood_support_embeddings,
                                                 eval_labels=None,
                                                 split_label="OOD_support",
                                                 return_exemplar_vectors=True,
                                                 dataset_q=None,
                                                 dataset_distance_quantile_per_class=None
                                                 )
            ood_support_predicted_labels = torch.argmax(ood_support_batch_f_positive_outputs, dim=1).detach().numpy().tolist()
            _, calibration_top_k_distances, calibration_top_k_distances_idx = \
                model.construct_support_index(support_exemplar_vectors_numpy=train_exemplar_vectors.detach().numpy(),
                                              calibration_exemplar_vectors_numpy=calibration_exemplar_vectors.detach().numpy(),
                                              ood_support_exemplar_vectors_numpy=ood_support_exemplar_vectors.detach().numpy(),
                                              ood_support_labels=ood_support_labels,
                                              ood_support_predicted_labels=ood_support_predicted_labels,
                                              ood_support_document_ids=ood_support_document_ids
                                              )
        else:
            # Set the exemplar vectors of training as the support set and fetch the calibration distances
            _, calibration_top_k_distances, calibration_top_k_distances_idx = \
                model.construct_support_index(support_exemplar_vectors_numpy=train_exemplar_vectors.detach().numpy(),
                                              calibration_exemplar_vectors_numpy=calibration_exemplar_vectors.detach().numpy())

        # Fetch the training distances. This will include the identity match, which is handled below.
        # Currently, we assume there are not duplicates in the data splits.
        train_top_k_distances__including_self, train_top_k_distances_idx__including_self = \
            model.get_top_support_distances(train_exemplar_vectors.detach().numpy())

        # get q values and dCDF for training; is_training_support=True will discard the first (identity) match
        # Note that the distance quantiles for training are determined by distances over training. The class
        # attribute model.trueClass_To_dCDF is over calibration, which is what should be used for new, unseen
        # test instances.
        train_dataset_q_values, train_trueClass_To_dataset_total_q_ood, train_trueClass_To_total_labels, \
            train_dataset_d0_values, train_trueClass_To_dCDF = model.set_summary_stats_for_support(
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
        calibration_dataset_q_values, calibration_trueClass_To_dataset_total_q_ood, calibration_trueClass_To_total_labels, calibration_dataset_d0_values, _ = model.set_summary_stats_for_support(calibration_exemplar_vectors.shape[0],
                                                                              calibration_top_k_distances,
                                                                              calibration_top_k_distances_idx,
                                                                              calibration_batch_f_positive_outputs,
                                                                              calibration_labels,
                                                                              is_training_support=False)
        for class_i in range(model.numberOfClasses):
            if len(model.trueClass_To_dCDF[class_i]) > 0:
                print(f"\tDistances: {constants.SPLIT_LABEL_calibration_during_training}: (class {class_i}) mean d0: {np.mean(model.trueClass_To_dCDF[class_i])}; "
                      f"median d0: {np.median(model.trueClass_To_dCDF[class_i])}, "
                      f"min: {np.min(model.trueClass_To_dCDF[class_i])}, "
                      f"max: {np.max(model.trueClass_To_dCDF[class_i])}, "
                      f"out of {len(model.trueClass_To_dCDF[class_i])}")
            else:
                print(
                    f"\tDistances: {constants.SPLIT_LABEL_calibration_during_training}: (class {class_i}): WARNING NO DISTANCES AVAILABLE")
        for class_i in range(model.numberOfClasses):
            print(f"\tTotal OOD q values (q<={model.ood_limit}): {constants.SPLIT_LABEL_calibration_during_training}: (class {class_i}): "
                  f"{calibration_trueClass_To_dataset_total_q_ood[class_i]} "
                  f"out of {calibration_trueClass_To_total_labels[class_i]}: "
                  f"{calibration_trueClass_To_dataset_total_q_ood[class_i]/(float(calibration_trueClass_To_total_labels[class_i]) if calibration_trueClass_To_total_labels[class_i] > 0 else 1.0)}")
        # get training distance quantiles, using distance empirical CDF over training
        train_dataset_distance_quantile_per_class = model.get_distance_quantiles(train_dataset_d0_values,
                                                                                 train_trueClass_To_dCDF=train_trueClass_To_dCDF)
        # get calibration training quantiles, using distance empirical CDF over calibration
        calibration_dataset_distance_quantile_per_class = model.get_distance_quantiles(calibration_dataset_d0_values,
                                                                                 train_trueClass_To_dCDF=None)

        if kUSE_STANDARD_CROSS_ENTROPY:
            assert False, "Comment this if standard cross entropy is desired"
            print(f">>Using standard cross entropy loss<<")
            # TEMP STANDARD CROSS ENTROPY:
            train_dataset_q_values = None
            train_dataset_distance_quantile_per_class = None
            calibration_dataset_q_values = None
            calibration_dataset_distance_quantile_per_class = None
            ## END STANDARD CROSS ENTROPY

        # additional eval passes to get rescaled and calculate metrics
        # train_dataset_q_values will change if predictions flip after rescaling
        train_acc, train_dataset_q_values, train_batch_f_positive_outputs, _, \
            train_balanced_accuracy, train_balanced_q_median, train_balanced_loss = \
            utils_classification.global_eval(options, model, eval_embeddings=train_embeddings, eval_labels=train_labels,
                                             split_label="TRAIN", return_exemplar_vectors=False,
                                             dataset_q=train_dataset_q_values,
                                             dataset_distance_quantile_per_class=train_dataset_distance_quantile_per_class
                                             )
        calibration_acc, _, _, _, \
            calibration_balanced_accuracy, calibration_balanced_q_median, calibration_balanced_loss = \
            utils_classification.global_eval(options, model, eval_embeddings=calibration_embeddings,
                                             eval_labels=calibration_labels,
                                             split_label=constants.SPLIT_LABEL_calibration_during_training, return_exemplar_vectors=False,
                                             dataset_q=calibration_dataset_q_values,
                                             dataset_distance_quantile_per_class=calibration_dataset_distance_quantile_per_class,
                                             set_model_unrescaledOutputCDF=True
                                             )
        print(f"Epoch: {e + 1} / Calibration set Accuracy: {calibration_acc}; "
              f"Calibration set Balanced Accuracy: {calibration_balanced_accuracy}; "
              f"Balanced median q: {calibration_balanced_q_median}, Balanced SDM loss: {calibration_balanced_loss}")

        if use_balanced_accuracy:
            is_best_running_epoch = calibration_balanced_accuracy >= max_dev_balanced_acc
        elif options.use_balanced_median_q:
            is_best_running_epoch = calibration_balanced_q_median >= max_dev_balanced_median_q
        else:
            is_best_running_epoch = calibration_balanced_loss <= min_dev_balanced_sdm_loss

        if calibration_balanced_loss <= min_dev_balanced_sdm_loss:
            min_dev_balanced_sdm_loss = calibration_balanced_loss
            min_dev_balanced_sdm_loss_epoch = e + 1
            train_balanced_sdm_loss_for_min_dev_sdm_loss = train_balanced_loss

        if calibration_acc >= max_dev_acc:
            max_dev_acc = calibration_acc
            max_dev_acc_epoch = e + 1
            train_acc_for_max_dev_acc = train_acc

        if calibration_balanced_accuracy >= max_dev_balanced_acc:
            max_dev_balanced_acc = calibration_balanced_accuracy
            max_dev_balanced_acc_epoch = e + 1
            train_balanced_acc_for_max_dev_acc = train_balanced_accuracy

        if calibration_balanced_q_median >= max_dev_balanced_median_q:
            max_dev_balanced_median_q = calibration_balanced_q_median
            max_dev_balanced_median_q_epoch = e + 1
            train_balanced_median_q_for_max_dev_median_q = train_balanced_q_median

        if is_best_running_epoch:
            model.increment_model_calibration_training_stage(set_value=1)
            utils_model.save_model(model, model_dir)
            logger.info(f"Model saved to model_dir")

        print(f"\tCurrent max Calibration set accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch} "
              f"(corresponding Training set accuracy: {train_acc_for_max_dev_acc})")
        print(f"\tCurrent max Calibration set Balanced accuracy: {max_dev_balanced_acc} at epoch {max_dev_balanced_acc_epoch} "
              f"(corresponding Training set Balanced accuracy: {train_balanced_acc_for_max_dev_acc})")
        print(f"\tCurrent max Calibration set Balanced MEDIAN Q: {max_dev_balanced_median_q} at epoch {max_dev_balanced_median_q_epoch} "
              f"(corresponding Training set Balanced MEDIAN Q: {train_balanced_median_q_for_max_dev_median_q})")
        print(f"\tCurrent min Calibration set Balanced SDM loss: {min_dev_balanced_sdm_loss} at epoch {min_dev_balanced_sdm_loss_epoch} "
              f"(corresponding Training set Balanced SDM loss: {train_balanced_sdm_loss_for_min_dev_sdm_loss})")

    print(f"\tMax Calibration set accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch} "
          f"(corresponding Training set accuracy: {train_acc_for_max_dev_acc})")
    print(
        f"\tMax Calibration set Balanced accuracy: {max_dev_balanced_acc} at epoch {max_dev_balanced_acc_epoch} "
        f"(corresponding Training set Balanced accuracy: {train_balanced_acc_for_max_dev_acc})")
    print(
        f"\tMax Calibration set Balanced MEDIAN Q: {max_dev_balanced_median_q} at epoch {max_dev_balanced_median_q_epoch} "
        f"(corresponding Training set Balanced MEDIAN Q: {train_balanced_median_q_for_max_dev_median_q}")
    print(
        f"\tMin Calibration set Balanced SDM loss: {min_dev_balanced_sdm_loss} at epoch {min_dev_balanced_sdm_loss_epoch} "
        f"(corresponding Training set Balanced SDM loss: {train_balanced_sdm_loss_for_min_dev_sdm_loss}")

    if use_balanced_accuracy:
        print(f"Final epoch chosen based on Balanced Accuracy.")
    elif options.use_balanced_median_q:
        print(f"Final epoch chosen based on Balanced MEDIAN Q.")
    else:
        print(f"Final epoch chosen based on the minimum Balanced SDM loss (over calibration).")

    print(f"Reloading best epoch model for training the model re-scaler")
    min_valid_qbin_for_class_conditional_accuracy, predicted_class_to_bin_to_median_output_magnitude = \
        train_rescaler(options, model_dir=model_dir)
    return max_dev_balanced_acc, max_dev_balanced_median_q, min_dev_balanced_sdm_loss, \
        min_valid_qbin_for_class_conditional_accuracy, \
        predicted_class_to_bin_to_median_output_magnitude


def train_rescaler(options, model_dir=None):
    assert model_dir is not None
    model = utils_model.load_model_torch(model_dir, torch.device("cpu"))
    if model.alpha != options.alpha:
        print(f">>Updating alpha from {model.alpha} (saved with the model) to {options.alpha} based on the "
              f"provided input arguments. However, note that the global statistics (across iterations) will "
              f"not be updated. Only use this option for debugging, and re-run the full iterations for deployment.<<")
        model.alpha = options.alpha
    if not options.only_update_rescaler_alpha:
        model = _train_model_rescaler(options, model, model.calibration_unrescaled_CDFquantiles,
                                      model.calibration_soft_qbins, model.calibration_labels)

        model.increment_model_calibration_training_stage(set_value=2)
        utils_model.save_model(model, model_dir)
        logger.info(f"Model saved to model_dir with trained rescaler")
        # reload in preparation for determining the bin threshold
        model = utils_model.load_model_torch(model_dir, torch.device("cpu"))

    predicted_class_to_bin_to_median_output_magnitude = model.set_class_conditional_non_ood_threshold(model.calibration_predicted_labels,
                                                  model.calibration_unrescaled_CDFquantiles,
                                                  model.calibration_soft_qbins,
                                                  model.calibration_labels)
    utils_model.save_model(model, model_dir)
    logger.info(f"Model saved to model_dir with training complete. Ready for testing.")
    return model.min_valid_qbin_for_class_conditional_accuracy, predicted_class_to_bin_to_median_output_magnitude


def _train_model_rescaler(options, model, calibration_unrescaled_CDFquantiles, calibration_soft_qbins, true_labels):
    start_time = time.time()
    # main_device = torch.device(options.main_device)
    # aux_device = torch.device(options.aux_device)
    # if main_device != aux_device:
    #     calibration_unrescaled_CDFquantiles, calibration_soft_qbins, true_labels = \
    #         calibration_unrescaled_CDFquantiles.to(aux_device), calibration_soft_qbins.to(aux_device), true_labels.to(aux_device)
    #     model.model_rescaler = model.model_rescaler.to(aux_device)
    # print(f"Training rescaler on {aux_device}")

    epochs = options.model_rescaler_training_max_epochs
    learning_rate = options.model_rescaler_training_learning_rate
    criterion = nn.NLLLoss()
    cumulative_losses = []
    parameters = filter(lambda p: p.requires_grad, model.model_rescaler.parameters())

    number_epochs_with_non_decreasing_loss_stopping_criteria = 10
    consecutive_epochs_with_non_decreasing_loss = 0
    best_rescaler_weights = None
    min_loss = torch.inf
    min_loss_epoch = -1

    optimizer = optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    for epoch in range(epochs):
        cumulative_losses_epoch = []
        shuffled_train_indexes = torch.randperm(calibration_unrescaled_CDFquantiles.shape[0])
        shuffled_calibration_unrescaled_CDFquantiles = calibration_unrescaled_CDFquantiles[shuffled_train_indexes]
        shuffled_calibration_soft_qbins = calibration_soft_qbins[shuffled_train_indexes]
        shuffled_true_labels = true_labels[shuffled_train_indexes]
        for unrescaled_CDFquantiles, soft_qbin, true_label in zip(shuffled_calibration_unrescaled_CDFquantiles,
                                                                  shuffled_calibration_soft_qbins,
                                                                  shuffled_true_labels):
            optimizer.zero_grad()
            model.train()
            loss = criterion(model(unrescaled_CDFquantiles.unsqueeze(0), soft_qbin.unsqueeze(1),
                  forward_type=constants.FORWARD_TYPE_TRAIN_RESCALER, train=True), true_label.long().unsqueeze(0))
            cumulative_losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
        mean_loss_for_epoch = np.mean(cumulative_losses_epoch)
        if mean_loss_for_epoch < min_loss:
            min_loss = mean_loss_for_epoch
            min_loss_epoch = epoch
            best_rescaler_weights = model.model_rescaler.weight.detach().clone()
        if epoch > 0:
            if mean_loss_for_epoch > min_loss:
                consecutive_epochs_with_non_decreasing_loss += 1
                print(f"\t -loss increased-")
                if consecutive_epochs_with_non_decreasing_loss > number_epochs_with_non_decreasing_loss_stopping_criteria:
                    break
            else:
                consecutive_epochs_with_non_decreasing_loss = 0
        cumulative_losses.append(mean_loss_for_epoch)

        print(f"Training model rescaler: Epoch: {epoch+1}: mean loss for epoch: {mean_loss_for_epoch}")
        print(f"\t Cumulative losses: {np.mean(cumulative_losses)}")
    print(f"Final epoch (min loss): {min_loss_epoch} with loss of {min_loss}")
    model.model_rescaler.weight = nn.Parameter(best_rescaler_weights)  # .to(main_device)
    print(f"Final rescaler weights: {model.model_rescaler.weight}")
    print(f"Total time to rescale: {time.time() - start_time}")
    return model
