# Copyright Reexpress AI, Inc. All rights reserved.

import torch

def assign_binary_reward_label(reference_document_string: str, generated_sentence: str) -> int:
    # Check for exact match, after removing whitespace
    return int(reference_document_string.strip().split() == generated_sentence.strip().split())


def generation_follows_formatting(reconstructed_document_string_with_tags: str,
                                  generated_sentence_with_tags: str) -> bool:
    return reconstructed_document_string_with_tags.strip() == generated_sentence_with_tags.strip()


def is_exact_match_excluding_boundary_whitespace(reference_document_string: str, generated_sentence: str) -> int:
    # Check for exact match, after removing whitespace
    return reference_document_string.strip() == generated_sentence.strip()


def get_sdm_outputs_for_index(sdm_outputs, index_tensor,
                              return_tensors_on_cpu=True,
                              keepdim=False):
    """
    Batch retrieve SDM outputs for the provided indexes.
    """
    # sdm_outputs and index_tensor should be on the same device.
    # sdm_outputs has shape: torch.Size([batch_size, class_size])
    # index_tensor has shape: torch.Size([batch_size])
    # Set keep_dims=True to return the values as torch.Size([batch_size, 1])

    # Extract the applicable index for each document from the SDM outputs using advanced indexing
    # Create indices for gathering the correct SDM output values
    batch_indices = torch.arange(len(index_tensor))
    sdm_outputs_for_index = sdm_outputs[batch_indices, index_tensor]

    if keepdim:
        sdm_outputs_for_index = sdm_outputs_for_index.unsqueeze(1)
    if return_tensors_on_cpu:
        return sdm_outputs_for_index.detach().cpu()
    else:
        return sdm_outputs_for_index
