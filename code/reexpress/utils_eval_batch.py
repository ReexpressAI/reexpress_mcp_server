# Copyright Reexpress AI, Inc. All rights reserved.

import constants

import torch
import torch.nn as nn


# When used with an SDM language model, use is_training_support=True to get the output over the training split.
# In that case, the model needs to be initialized with is_sdm_network_verification_layer=True so that
# self.train_trueClass_To_dCDF is available on reload (and the model must be loaded with
# load_for_inference=False).
def get_q_and_d_from_embeddings(model, eval_batch_size, eval_embeddings,
                                main_device, is_training_support=False, return_exemplar_vectors=False):
    if is_training_support:
        assert model.is_sdm_network_verification_layer
        assert len(model.train_trueClass_To_dCDF) > 0
    current_device = main_device
    model = model.to(current_device)

    # First get predictions and exemplar vectors.
    eval_batch_f_outputs, eval_exemplar_vectors = \
        get_predictions_and_exemplar_vectors(eval_batch_size, model,
                                             eval_embeddings,
                                             current_device,
                                             place_output_on_cpu=True)
    return model.get_q_and_d_from_exemplars(batch_f=eval_batch_f_outputs,
                                            exemplar_vectors=eval_exemplar_vectors,
                                            is_training_support=is_training_support,
                                            return_exemplar_vectors=return_exemplar_vectors)


def print_metrics(e, numberOfClasses, split_label_name, per_class_loss_as_list, balanced_loss, marginal_loss,
                  per_class_accuracy_as_list, balanced_accuracy, marginal_accuracy,
                  per_class_q_as_list, balanced_q, marginal_q):
    print(f"---{split_label_name}---")
    if e > -1:  # For use during training
        print(f"Epoch: {e + 1} / {split_label_name} Balanced SDM Loss: {balanced_loss}")
    else:
        print(f"Balanced SDM Loss: {balanced_loss}")
    print(f"{split_label_name} Marginal SDM Loss: {marginal_loss}")
    print(f"{split_label_name} SDM Loss by class:")
    for class_label in range(numberOfClasses):
        print(f"\tClass {class_label}: {per_class_loss_as_list[class_label]}")
    print(f"{split_label_name} Marginal Accuracy: {marginal_accuracy}; Balanced Accuracy: {balanced_accuracy}")
    print(f"{split_label_name} Accuracy by class:")
    for class_label in range(numberOfClasses):
        print(f"\tClass {class_label}: {per_class_accuracy_as_list[class_label]}")
    print(f"{split_label_name} Marginal mean q: {marginal_q}; Balanced mean q: {balanced_q}")
    print(f"{split_label_name} mean q by class:")
    for class_label in range(numberOfClasses):
        print(f"\tClass {class_label}: {per_class_q_as_list[class_label]}")


def get_metrics_from_cached_outputs(batch_size, model,
                                    cached_f_outputs,
                                    current_device,
                                    held_out_labels,
                                    q_values=None,
                                    distance_quantile_per_class=None):
    # This calculates the metrics from cached_f_outputs. The convolution is not rerun.
    criterion = nn.NLLLoss(reduction="none")
    if q_values is None:
        q_values = torch.zeros(cached_f_outputs.shape[0], 1) + (torch.e - model.q_rescale_offset)
    held_out_size = cached_f_outputs.shape[0]
    batch_num = 0
    model.eval()

    class_size = model.numberOfClasses
    running_class_counts = torch.zeros(class_size).to(current_device)
    running_loss_sum_by_class = torch.zeros(class_size).to(current_device)
    running_is_correct_sum_by_class = torch.zeros(class_size).to(current_device)
    running_q_sum_by_class = torch.zeros(class_size).to(current_device)
    all_sdm_outputs = []

    # marginal values
    running_total_loss = 0.0
    running_total_correct = 0
    running_total_q = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, held_out_size, batch_size):
            batch_num += 1
            batch_range = min(batch_size, held_out_size - i)

            batch_f = cached_f_outputs[i:i + batch_range].to(current_device)
            batch_y = held_out_labels[i:i + batch_range].to(current_device)
            batch_q = q_values[i:i + batch_range].to(current_device)
            if distance_quantile_per_class is None:
                batch_distance_quantile_per_class = None
            else:
                batch_distance_quantile_per_class = distance_quantile_per_class[i:i + batch_range].to(current_device)

            batch_sdm_log = \
                model.soft_sdm_max(batch_f, batch_q,
                                   distance_quantile_per_class=batch_distance_quantile_per_class,
                                   log=True, change_of_base=True)
            batch_sdm = \
                model.soft_sdm_max_log_to_probability(batch_sdm_log, batch_q)
            # Reference: The use of model.soft_sdm_max_log_to_probability() above is equivalent to the following, but
            # computationally cheaper given batch_sdm_log:
            # batch_sdm = \
            #     model.soft_sdm_max(batch_f, batch_q,
            #                        distance_quantile_per_class=batch_distance_quantile_per_class,
            #                        log=False, change_of_base=True)
            if len(batch_f.shape) == 1:
                batch_sdm_log = batch_sdm_log.unsqueeze(0)
                batch_sdm = batch_sdm.unsqueeze(0)

            all_sdm_outputs.append(batch_sdm.detach().cpu())
            batch_predictions = torch.argmax(batch_f, dim=1)

            # Calculate correct predictions per class
            is_correct = (batch_predictions == batch_y).float()
            running_is_correct_sum_by_class += torch.zeros(class_size,
                                                           device=current_device).scatter_add_(0,
                                                                                               batch_y,
                                                                                               is_correct)
            # Calculate loss per class
            loss = criterion(batch_sdm_log, batch_y)
            running_loss_sum_by_class += torch.zeros(class_size,
                                                     device=current_device).scatter_add_(0, batch_y, loss)
            # Accumulate q_values per class
            batch_q_squeezed = batch_q.squeeze(-1) if batch_q.dim() > 1 else batch_q
            running_q_sum_by_class += torch.zeros(class_size,
                                                  device=current_device).scatter_add_(0,
                                                                                      batch_y,
                                                                                      batch_q_squeezed)
            # Update class counts
            running_class_counts += torch.zeros(class_size,
                                                device=current_device).scatter_add_(0,
                                                                                    batch_y,
                                                                                    torch.ones_like(batch_y,
                                                                                                    device=current_device,
                                                                                                    dtype=torch.float))
            # Accumulate overall (marginal) metrics
            running_total_loss += loss.sum().item()
            running_total_correct += is_correct.sum().item()
            running_total_q += batch_q_squeezed.sum().item()
            total_samples += batch_f.shape[0]

    # Calculate per-class average loss
    per_class_avg_loss = torch.where(running_class_counts > 0,
                                     running_loss_sum_by_class / running_class_counts,
                                     torch.zeros_like(running_loss_sum_by_class, device=current_device))
    balanced_loss = per_class_avg_loss.mean()
    per_class_loss_as_list = [float(x) for x in per_class_avg_loss.detach().cpu().numpy().tolist()]

    # Calculate per-class accuracy
    per_class_accuracy = torch.where(running_class_counts > 0,
                                     running_is_correct_sum_by_class / running_class_counts,
                                     torch.zeros_like(running_is_correct_sum_by_class, device=current_device))
    balanced_accuracy = per_class_accuracy.mean()
    per_class_accuracy_as_list = [float(x) for x in per_class_accuracy.detach().cpu().numpy().tolist()]

    # Calculate per-class average q_values
    per_class_avg_q = torch.where(running_class_counts > 0,
                                  running_q_sum_by_class / running_class_counts,
                                  torch.zeros_like(running_q_sum_by_class, device=current_device))
    balanced_q = per_class_avg_q.mean()
    per_class_q_as_list = [float(x) for x in per_class_avg_q.detach().cpu().numpy().tolist()]

    # Calculate overall (marginal) metrics
    marginal_loss = running_total_loss / total_samples if total_samples > 0 else 0.0
    marginal_accuracy = running_total_correct / total_samples if total_samples > 0 else 0.0
    marginal_q = running_total_q / total_samples if total_samples > 0 else 0.0

    return (per_class_loss_as_list, balanced_loss.item(), marginal_loss,
            per_class_accuracy_as_list, balanced_accuracy.item(), marginal_accuracy,
            per_class_q_as_list, balanced_q.item(), marginal_q,
            torch.cat(all_sdm_outputs, dim=0))


def get_predictions_and_exemplar_vectors(batch_size, model,
                                         held_out_embeddings,
                                         current_device,
                                         place_output_on_cpu=True):

    q_values = torch.zeros(held_out_embeddings.shape[0], 1) + (torch.e - model.q_rescale_offset)
    held_out_size = held_out_embeddings.shape[0]
    batch_num = 0
    model.eval()

    all_f_outputs = []
    all_exemplar_vectors = []
    with torch.no_grad():
        for i in range(0, held_out_size, batch_size):
            batch_num += 1
            batch_range = min(batch_size, held_out_size - i)

            batch_x = held_out_embeddings[i:i + batch_range].to(current_device)
            batch_q = q_values[i:i + batch_range].to(current_device)
            batch_distance_quantile_per_class = None
            batch_f, _, batch_exemplar_vectors = model(batch_x, batch_q,
                                                       batch_distance_quantile_per_class=batch_distance_quantile_per_class,
                                                       forward_type=constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS,
                                                       train=False)

            if len(batch_f.shape) == 1:
                batch_f = batch_f.unsqueeze(0)
                batch_exemplar_vectors = batch_exemplar_vectors.unsqueeze(0)
            if place_output_on_cpu:
                all_f_outputs.append(batch_f.detach().cpu())
                all_exemplar_vectors.append(batch_exemplar_vectors.detach().cpu())
            else:
                all_f_outputs.append(batch_f.detach())
                all_exemplar_vectors.append(batch_exemplar_vectors.detach())
    return torch.cat(all_f_outputs, dim=0), torch.cat(all_exemplar_vectors, dim=0)
