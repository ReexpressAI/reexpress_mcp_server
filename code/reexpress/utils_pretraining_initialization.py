# Copyright Reexpress AI, Inc. All rights reserved.

import constants
import utils_model

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np


def load_pretraining_initialization_tensors(pretraining_initialization_tensors_file):
    concatenated_embeddings_and_labels = torch.load(
        pretraining_initialization_tensors_file,
        weights_only=True, map_location=torch.device("cpu"))
    # concatenated_embeddings_and_labels is shape: [total_instances BY (embedding dimension + labels)]
    train_embeddings = concatenated_embeddings_and_labels[:, 0:-1]
    train_labels = concatenated_embeddings_and_labels[:, -1].long()
    return train_embeddings, train_labels


def pretrain(options, model=None, model_dir=None,
             held_out_embeddings=None,
             held_out_labels=None):
    # currently this runs on options.aux_device
    current_device = torch.device(options.aux_device)
    total_epochs = options.pretraining_initialization_epochs
    print(f"Pretraining initialization for {total_epochs} epochs on {current_device}")
    assert model is not None
    model = model.to(current_device)
    if model.is_gen_ai:
        print(f"FREEZING LLM DISTRIBUTION WEIGHTS")
        model.fc_negative.weight.requires_grad = False
        model.fc_positive.weight.requires_grad = False
        model.fc_original.weight.requires_grad = False

    train_embeddings, train_labels = \
        load_pretraining_initialization_tensors(options.pretraining_initialization_tensors_file)
    train_size = train_embeddings.shape[0]

    print(f"Starting pretraining over {train_size} instances")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=options.pretraining_learning_rate, betas=(0.9, 0.999), eps=1e-08)

    criterion = nn.NLLLoss()
    all_epoch_cumulative_losses = []

    min_held_out_balanced_loss = np.inf
    min_held_out_balanced_loss_epoch = -1

    best_model_conv_weight = None
    best_model_conv_bias = None
    best_model_fc_weight = None
    best_model_fc_bias = None

    batch_size = options.batch_size
    default_training_q_values = torch.zeros(train_embeddings.shape[0], 1) + (np.e - model.q_rescale_offset)
    for e in range(total_epochs):
        # shuffle data
        shuffled_train_indexes = torch.randperm(train_embeddings.shape[0])
        shuffled_train_embeddings = train_embeddings[shuffled_train_indexes]
        shuffled_train_labels = train_labels[shuffled_train_indexes]
        shuffled_q = default_training_q_values[shuffled_train_indexes]
        batch_num = 0
        cumulative_losses = []

        for i in range(0, train_size, batch_size):
            batch_num += 1
            batch_range = min(batch_size, train_size - i)

            batch_x = shuffled_train_embeddings[i:i + batch_range].to(current_device)
            batch_y = shuffled_train_labels[i:i + batch_range].to(current_device)
            batch_q = shuffled_q[i:i + batch_range].to(current_device)
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

        print(f"---------------Pretraining Epoch: {e + 1}---------------")
        print(f"Pretraining Epoch average loss (over-pretraining set): {np.mean(cumulative_losses)}")
        all_epoch_cumulative_losses.extend(cumulative_losses)
        print(f"Pretraining Average loss across all mini-batches (all epochs, over-pretraining set): "
              f"{np.mean(all_epoch_cumulative_losses)}")

        held_out_loss_by_class_list, held_out_balanced_loss = get_loss_over_heldout_data(options, model,
                                                                                         held_out_embeddings,
                                                                                         held_out_labels,
                                                                                         current_device)
        print(f"Pretraining Epoch: {e + 1} / Held-out set Balanced loss: {held_out_balanced_loss}")
        print(f"Pretraining Epoch: {e + 1} / Held-out set Balanced loss by class:")
        for class_label in range(model.numberOfClasses):
            print(f"\tClass {class_label}: {held_out_loss_by_class_list[class_label]}")
        is_best_running_epoch = held_out_balanced_loss <= min_held_out_balanced_loss
        if held_out_balanced_loss <= min_held_out_balanced_loss:
            min_held_out_balanced_loss = held_out_balanced_loss
            min_held_out_balanced_loss_epoch = e + 1
        if is_best_running_epoch and total_epochs > 1:
            # Here, we are only updating the adaptor layer. The summary statistics and other data structures
            # have not (necessarily) yet been created. This pretraining is overall typically very fast,
            # so we simply cache the weights to memory.
            best_model_conv_weight = model.conv.weight.detach().clone()
            best_model_conv_bias = model.conv.bias.detach().clone()
            best_model_fc_weight = model.fc.weight.detach().clone()
            best_model_fc_bias = model.fc.bias.detach().clone()
            print(f">Cached best epoch parameters<")
        print(f"\tPretraining: Current min held-out set Balanced loss: "
              f"{min_held_out_balanced_loss} at epoch {min_held_out_balanced_loss_epoch}")
    print(
        f"\tPretraining: Min held-out set Balanced loss: "
        f"{min_held_out_balanced_loss} at epoch {min_held_out_balanced_loss_epoch}")

    if total_epochs > 1:
        # Add best weights
        model.conv.weight = nn.Parameter(best_model_conv_weight)
        model.conv.bias = nn.Parameter(best_model_conv_bias)
        model.fc.weight = nn.Parameter(best_model_fc_weight)
        model.fc.bias = nn.Parameter(best_model_fc_bias)
    return model


def get_loss_over_heldout_data(options, model,
                               held_out_embeddings,
                               held_out_labels,
                               current_device):
    if current_device == torch.device('mps'):
        # The scatter operations are not currently implemented on mps, so we need to move to cpu for the time being:
        current_device = torch.device('cpu')
    criterion = nn.NLLLoss(reduction="none")
    batch_size = options.batch_size
    default_training_q_values = torch.zeros(held_out_embeddings.shape[0], 1) + (np.e - model.q_rescale_offset)
    held_out_size = held_out_embeddings.shape[0]
    batch_num = 0
    model.eval()

    class_size = model.numberOfClasses
    running_class_counts = torch.zeros(class_size).to(current_device)
    running_loss_sum_by_class = torch.zeros(class_size).to(current_device)

    with torch.no_grad():
        for i in range(0, held_out_size, batch_size):
            batch_num += 1
            batch_range = min(batch_size, held_out_size - i)

            batch_x = held_out_embeddings[i:i + batch_range].to(current_device)
            batch_y = held_out_labels[i:i + batch_range].to(current_device)
            batch_q = default_training_q_values[i:i + batch_range].to(current_device)
            batch_distance_quantile_per_class = None
            _, rescaled_batch_output = model(batch_x, batch_q,
                                             batch_distance_quantile_per_class=batch_distance_quantile_per_class,
                                             forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION,
                                             train=True)
            if len(rescaled_batch_output.shape) == 1:
                loss = criterion(rescaled_batch_output.unsqueeze(0), batch_y)
            else:
                loss = criterion(rescaled_batch_output, batch_y)

            running_loss_sum_by_class += torch.zeros(class_size,
                                                     device=current_device).scatter_reduce_(0,
                                                                                            batch_y,
                                                                                            loss,
                                                                                            reduce='sum')
            running_class_counts += torch.zeros(class_size,
                                                device=current_device).scatter_add_(0,
                                                                                    batch_y,
                                                                                    torch.ones_like(batch_y,
                                                                                                    device=current_device,
                                                                                                    dtype=torch.float))

    per_class_avg = torch.where(running_class_counts > 0,
                                running_loss_sum_by_class / running_class_counts,
                                torch.zeros_like(running_loss_sum_by_class, device=current_device))
    balanced_loss = per_class_avg.mean()
    return [float(x) for x in per_class_avg.detach().cpu().numpy().tolist()], balanced_loss.item()
