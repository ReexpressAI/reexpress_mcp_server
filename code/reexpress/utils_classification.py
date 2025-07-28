# Copyright Reexpress AI, Inc. All rights reserved.

import constants
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import argparse
import copy

import math


# def get_max_bifurcated_distribution(vocab, expanded_distribution):
#     max_negative = torch.amax(expanded_distribution[:, 0: vocab], dim=1, keepdims=True)
#     max_positive = torch.amax(expanded_distribution[:, vocab:], dim=1, keepdims=True)
#     return torch.cat([max_negative, max_positive], dim=1)
#
# def transform_gen_ai_output_to_binary_classification(gen_ai_vocab, batch_f_positive, soft_sdm_max_unrescaled_batch_output):
#     return get_max_bifurcated_distribution(gen_ai_vocab, batch_f_positive), \
#         get_max_bifurcated_distribution(gen_ai_vocab, soft_sdm_max_unrescaled_batch_output)


def global_eval(options, model, eval_embeddings=None, eval_labels=None, split_label=None,
                return_exemplar_vectors=False,
                dataset_q=None,
                dataset_distance_quantile_per_class=None, set_model_unrescaledOutputCDF=False, main_device=None,
                end_of_document_indicators=None): #, construct_binary_prediction_from_genai_dual_distribution=False):
    # if end_of_document_indicators is not None, we assume that the true label for the full document is equal to
    # the token level label

    if eval_labels is not None:
        print(f"++Evaluating model over {split_label}++")
    model.eval()
    criterion = nn.NLLLoss(reduction="none")
    eval_size = eval_embeddings.shape[0]
    if set_model_unrescaledOutputCDF:
        assert eval_labels is not None
        assert split_label == constants.SPLIT_LABEL_calibration_during_training
        model.trueClass_To_unrescaledOutputCDF = {}
        for trueLabel in range(model.numberOfClasses):
            model.trueClass_To_unrescaledOutputCDF[trueLabel] = []

    batch_size = options.eval_batch_size

    batch_num = 0
    cumulative_losses = []
    acc = []
    acc_by_class = {}
    loss = []
    loss_by_class = {}
    for class_i in range(model.numberOfClasses):
        acc_by_class[class_i] = []
        loss_by_class[class_i] = []
    batch_f_positive_outputs = []  # before re-scaling
    soft_sdm_max_batch_outputs = []
    updated_dataset_q = []
    total_q_flips = 0

    # document-level (when applicable)
    document_acc = []
    document_acc_by_class = {}
    for class_i in range(model.numberOfClasses):
        document_acc_by_class[class_i] = []

    if eval_labels is not None:
        updated_dataset_q_by_class = {}  # only used for stats, as the indexes are lost
        for label in range(model.numberOfClasses):
            updated_dataset_q_by_class[label] = []
    if return_exemplar_vectors:
        all_exemplar_vectors = []
    if dataset_q is None:
        # standard softmax:
        dataset_q = (torch.zeros(eval_size, 1) + (np.e - model.q_rescale_offset)).to(main_device)
    else:
        assert dataset_q.shape[1] == 1
        assert dataset_q.shape[0] == eval_embeddings.shape[0]
    if dataset_distance_quantile_per_class is not None:
        assert dataset_distance_quantile_per_class.shape[1] == model.numberOfClasses
        assert dataset_distance_quantile_per_class.shape[0] == eval_embeddings.shape[0]

    running_instance_i = 0
    with torch.no_grad():
        for i in range(0, eval_size, batch_size):
            # if batch_num % int(num_batch_instances * 0.25) == 0:
            #     print(f"Eval progress: {batch_num/num_batch_instances}")
            batch_num += 1
            batch_range = min(batch_size, eval_size - i)

            batch_x = eval_embeddings[i:i + batch_range]
            batch_q = dataset_q[i:i + batch_range]
            if dataset_distance_quantile_per_class is not None:
                batch_distance_quantile_per_class = dataset_distance_quantile_per_class[i:i + batch_range]
            else:
                batch_distance_quantile_per_class = None
            if eval_labels is not None:
                batch_y = eval_labels[i:i + batch_range]
            if return_exemplar_vectors:
                batch_f_positive, soft_sdm_max_unrescaled_batch_output, exemplar_vectors = model(batch_x, batch_q,
                                               batch_distance_quantile_per_class=batch_distance_quantile_per_class,
                                               forward_type=constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS,
                                               train=False)
                if len(exemplar_vectors.shape) == 1:
                    all_exemplar_vectors.append(exemplar_vectors.unsqueeze(0).cpu())
                else:
                    all_exemplar_vectors.append(exemplar_vectors.cpu())
            else:
                batch_f_positive, soft_sdm_max_unrescaled_batch_output = model(batch_x, batch_q,
                             batch_distance_quantile_per_class=batch_distance_quantile_per_class,
                             forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION,
                             train=False)

            if len(soft_sdm_max_unrescaled_batch_output.shape) == 1:
                soft_sdm_max_unrescaled_batch_output = soft_sdm_max_unrescaled_batch_output.unsqueeze(0)
            if len(batch_f_positive.shape) == 1:
                batch_f_positive = batch_f_positive.unsqueeze(0)
            # if construct_binary_prediction_from_genai_dual_distribution:
            #     batch_f_positive, soft_sdm_max_unrescaled_batch_output = \
            #         transform_gen_ai_output_to_binary_classification(model.gen_ai_vocab, batch_f_positive,
            #                                                          soft_sdm_max_unrescaled_batch_output)
            if eval_labels is not None:
                batch_updated_q_values = torch.zeros(soft_sdm_max_unrescaled_batch_output.shape[0], 1).to(main_device)
            for batch_i in range(soft_sdm_max_unrescaled_batch_output.shape[0]):
                soft_sdm_max_batch_outputs.append(soft_sdm_max_unrescaled_batch_output[batch_i].unsqueeze(0).cpu())  # soft_sdm_max, not logits
                batch_f_positive_outputs.append(batch_f_positive[batch_i].unsqueeze(0).cpu())  # logits
                sdm_pred_y = torch.argmax(soft_sdm_max_unrescaled_batch_output[batch_i], dim=0).item()
                # the convention is that the logits are used for overall accuracy; flips are unreliable predictions
                # and get handled w.r.t. uncertainty by low q values. In the current version, we use uniform
                # distance values, but a flip is possible in the cases when the output goes to parity (i.e., when the
                # input is, in effect, OOD) because the default index will be chosen in the argmax.
                logit_pred_y = torch.argmax(batch_f_positive[batch_i], dim=0).item()
                if sdm_pred_y != logit_pred_y:
                    # rescaling flipped the prediction, so set q = 0 for subsequent calculations
                    updated_dataset_q.append(0)
                    total_q_flips += 1
                    if eval_labels is not None:
                        batch_updated_q_values[batch_i, 0] = 0.0
                        updated_dataset_q_by_class[batch_y[batch_i].item()].append(0)
                else:
                    updated_dataset_q.append(batch_q[batch_i, 0].item())
                    if eval_labels is not None:
                        batch_updated_q_values[batch_i, 0] = batch_q[batch_i, 0]
                        updated_dataset_q_by_class[batch_y[batch_i].item()].append(batch_q[batch_i, 0].item())
                if eval_labels is not None:
                    true_y = batch_y[batch_i].item()
                    acc.append(logit_pred_y == true_y)
                    acc_by_class[true_y].append(logit_pred_y == true_y)
                    if end_of_document_indicators is not None and end_of_document_indicators[running_instance_i] == 1.0:
                        document_acc.append(logit_pred_y == true_y)
                        document_acc_by_class[true_y].append(logit_pred_y == true_y)
                    if set_model_unrescaledOutputCDF:
                        model.trueClass_To_unrescaledOutputCDF[true_y].append(soft_sdm_max_unrescaled_batch_output[batch_i][true_y].item())
                running_instance_i += 1
            if eval_labels is not None:
                # calculate loss using the updated q
                # Here, we do not need to rerun the convolution, so we use the cached version of batch_f_positive
                # from above and call model.soft_sdm_max() directly
                batch_soft_sdm_max_log_with_updated_q = \
                    model.soft_sdm_max(batch_f_positive, batch_updated_q_values,
                                       distance_quantile_per_class=batch_distance_quantile_per_class,
                                       log=True, change_of_base=True)
                batch_loss_no_reduction = criterion(batch_soft_sdm_max_log_with_updated_q, batch_y).cpu()
                loss.extend(batch_loss_no_reduction.tolist())
                for batch_i in range(soft_sdm_max_unrescaled_batch_output.shape[0]):
                    true_y = batch_y[batch_i].item()
                    loss_by_class[true_y].append(batch_loss_no_reduction[batch_i].item())

    if end_of_document_indicators:
        assert running_instance_i == len(end_of_document_indicators), f"ERROR: There is a mismatch with indexing to " \
                                                                      f"determine document-level labels. Evaluation " \
                                                                      f"will be incorrect."
    soft_sdm_max_batch_outputs = torch.cat(soft_sdm_max_batch_outputs, dim=0)
    batch_f_positive_outputs = torch.cat(batch_f_positive_outputs, dim=0)
    if eval_labels is not None:
        if set_model_unrescaledOutputCDF:
            for trueLabel in range(model.numberOfClasses):
                model.trueClass_To_unrescaledOutputCDF[trueLabel].sort()
            # another pass through calibration to set summary stats:
            calibration_unrescaled_CDFquantiles = []
            calibration_soft_qbins = []
            for cal_i in range(eval_size):
                unrescaled_CDFquantiles = torch.zeros(model.numberOfClasses)
                for label in range(model.numberOfClasses):
                    unrescaled_CDFquantiles[label] = model.getCDFIndex(model.trueClass_To_unrescaledOutputCDF,
                                                                       soft_sdm_max_batch_outputs[cal_i][label].item(),
                                                                       label,
                                                                       reverse=False,
                                                                       val_in_0to1=True)
                assert torch.argmax(batch_f_positive_outputs[cal_i]) == model.calibration_predicted_labels[cal_i], \
                "Error: There is an unexpected mismatch between the model's saved calibration predictions and " \
                "the argmax logits here."
                # uses updated q and logit prediction; this, in effect, handles flips:
                soft_qbin = (model.q_rescale_offset + updated_dataset_q[cal_i]) ** unrescaled_CDFquantiles[model.calibration_predicted_labels[cal_i].item()]
                soft_qbin = np.log(soft_qbin)
                calibration_unrescaled_CDFquantiles.append(unrescaled_CDFquantiles.unsqueeze(0))
                calibration_soft_qbins.append(torch.tensor([soft_qbin]).unsqueeze(1))
            model.calibration_unrescaled_CDFquantiles = torch.cat(calibration_unrescaled_CDFquantiles, dim=0)
            model.calibration_soft_qbins = torch.cat(calibration_soft_qbins, dim=0)

        mean_overall_acc = np.mean(acc)
        print(f"\t{split_label}: Total q reversions to 0: {total_q_flips}.")
        print(
            f"\t{split_label}: Marginal mean q: {np.mean(updated_dataset_q)}; "
            f"Marginal median q: {np.median(updated_dataset_q)}")
        balanced_q_median_list = []
        for class_i in range(model.numberOfClasses):
            if len(updated_dataset_q_by_class[class_i]) > 0:
                print(f"\t{split_label}: (class {class_i}) mean q: {np.mean(updated_dataset_q_by_class[class_i])}; "
                      f"median q: {np.median(updated_dataset_q_by_class[class_i])}, "
                      f"min: {np.min(updated_dataset_q_by_class[class_i])}, "
                      f"max: {np.max(updated_dataset_q_by_class[class_i])}, "
                      f"out of {len(updated_dataset_q_by_class[class_i])}")
                balanced_q_median_list.append(np.median(updated_dataset_q_by_class[class_i]))
            else:
                print(f"\t{split_label}: (class {class_i}): WARNING: There are no Similarity (q) values for this class. "
                      f"Setting the median q value to 0.")
                balanced_q_median_list.append(0)
        balanced_q_median = np.mean(balanced_q_median_list)  # mean of the median q's across classes
        print(f"\t==>{split_label}: Balanced (i.e., averaged over classes) median q: {balanced_q_median}")
        print(f"{split_label}: Overall ACC: {mean_overall_acc} out of {len(acc)}")
        balanced_accuracy_list = []
        for class_i in range(model.numberOfClasses):
            print(f"\t{split_label}: Acc (class {class_i}): {np.mean(acc_by_class[class_i])} out of {len(acc_by_class[class_i])}")
            balanced_accuracy_list.append(np.mean(acc_by_class[class_i]))
        balanced_accuracy = np.mean(balanced_accuracy_list)  # mean of the accuracies across classes
        print(f"\t..>{split_label}: Balanced (i.e., averaged over classes) accuracy: {balanced_accuracy}")

        print(f"///////{split_label} LOSS")
        balanced_loss_list = []
        for class_i in range(model.numberOfClasses):
            print(f"\t{split_label}: Avg. SDM loss (class {class_i}): {np.mean(loss_by_class[class_i])} out of {len(loss_by_class[class_i])}")
            balanced_loss_list.append(np.mean(loss_by_class[class_i]))
        balanced_loss = np.mean(balanced_loss_list)
        print(
            f"\t{split_label}: Avg. SDM loss: {np.mean(loss)}; "
            f"Median SDM loss: {np.median(loss)}")
        print(f"\t@@>{split_label} Balanced SDM loss: {balanced_loss}")

        if end_of_document_indicators is not None:
            document_level_mean_overall_acc = np.mean(document_acc)
            print(f"{split_label}: Overall document-level ACC: {document_level_mean_overall_acc} out of {len(document_acc)} documents")
            document_level_balanced_accuracy_list = []
            for class_i in range(model.numberOfClasses):
                print(f"\t{split_label}: Acc (class {class_i}): {np.mean(document_acc_by_class[class_i])} out of {len(document_acc_by_class[class_i])}")
                document_level_balanced_accuracy_list.append(np.mean(document_acc_by_class[class_i]))
            document_level_balanced_accuracy = np.mean(document_level_balanced_accuracy_list)  # mean of the accuracies across classes at the *document level*
            print(f"\t++>{split_label}: Document-level balanced (i.e., averaged over classes) accuracy: "
                  f"{document_level_balanced_accuracy}")

        if return_exemplar_vectors:
            return mean_overall_acc, torch.tensor(updated_dataset_q).unsqueeze(1).to(main_device), \
                batch_f_positive_outputs, soft_sdm_max_batch_outputs, \
                torch.cat(all_exemplar_vectors, dim=0), \
                balanced_accuracy, balanced_q_median, balanced_loss
        else:
            return mean_overall_acc, torch.tensor(updated_dataset_q).unsqueeze(1).to(main_device), \
                batch_f_positive_outputs, soft_sdm_max_batch_outputs, \
                balanced_accuracy, balanced_q_median, balanced_loss
    else:
        if return_exemplar_vectors:
            return torch.tensor(updated_dataset_q).unsqueeze(1).to(main_device), batch_f_positive_outputs, \
                soft_sdm_max_batch_outputs, torch.cat(all_exemplar_vectors, dim=0)
        else:
            return torch.tensor(updated_dataset_q).unsqueeze(1).to(main_device), batch_f_positive_outputs, \
                soft_sdm_max_batch_outputs

